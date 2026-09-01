package ollama

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"time"

	"digital.vasic.llmprovider/pkg/discovery"
	"digital.vasic.llmprovider/pkg/i18n"
	"digital.vasic.llmprovider/pkg/models"
	"digital.vasic.llmprovider/pkg/settings"
)

// Vendor-native environment aliases for this adapter's constructor defaults.
//
// The canonical names are this module's own — LLMPROVIDER_OLLAMA_BASE_URL,
// LLMPROVIDER_OLLAMA_MODEL, LLMPROVIDER_OLLAMA_TIMEOUT, resolved through
// pkg/settings, which also documents the precedence. The three below are
// ollama's OWN variable names — OLLAMA_HOST is the one the `ollama` CLI and
// server read — and are accepted as ALIASES, consulted after the canonical key.
// An operator who has already exported OLLAMA_HOST for the CLI therefore gets
// this library pointed at the same place without doing anything further, while
// this module's convention still wins wherever both are set.
//
// Each is consulted only when the caller passed the corresponding constructor
// argument empty, so an explicit argument always outranks the environment.
const (
	// EnvBaseURL is ollama's own endpoint variable.
	EnvBaseURL = "OLLAMA_HOST"
	// EnvModel is the vendor-shaped alias for the default model.
	EnvModel = "OLLAMA_MODEL"
	// EnvTimeout is the vendor-shaped alias for the default HTTP timeout. Its
	// value is parsed with time.ParseDuration, so "180s", "3m" and "1h30m" are
	// all accepted. A value that does not parse is IGNORED rather than silently
	// treated as zero — a zero http.Client.Timeout means "no timeout at all",
	// which is the opposite of what a mistyped timeout was reaching for.
	EnvTimeout = "OLLAMA_TIMEOUT"
)

// Constructor defaults, applied when neither an argument nor the environment
// supplies a value.
const (
	DefaultBaseURL = "http://localhost:11434"
	DefaultModel   = "llama2"
	// DefaultTimeout is generous because ollama loads the model on the first
	// request of a session. It is a DEFAULT, not a ceiling: see SetTimeout.
	DefaultTimeout = 120 * time.Second
)

// Keys read from models.ModelParameters.ProviderSpecific. That map is this
// module's documented seam for per-backend generation options — the perplexity
// and venice adapters read their own keys from it the same way — so ollama's
// options travel through it rather than through a widened LLMProvider
// interface, which module rule 1 forbids.
const (
	// ParamFormat carries ollama's `format` field: either the string "json",
	// or a JSON Schema document (map[string]any) for constrained decoding. A
	// schema makes the model's output parse-able by construction instead of
	// by hope, which is what a grounded answering pipeline needs.
	ParamFormat = "format"
	// ParamSeed carries ollama's `seed` option. With Temperature 0 a fixed
	// seed makes a run reproducible by a reviewer; a result nobody else can
	// reproduce is not evidence.
	ParamSeed = "seed"
	// ParamNumPredict carries ollama's `num_predict` option. It is an alias
	// for ModelParameters.MaxTokens; when both are set, this one wins, because
	// a provider-specific key is by definition the more specific instruction.
	ParamNumPredict = "num_predict"
)

// RetryConfig defines retry behavior for API calls
type RetryConfig struct {
	MaxRetries   int
	InitialDelay time.Duration
	MaxDelay     time.Duration
	Multiplier   float64
}

// DefaultRetryConfig returns sensible defaults for Ollama API retry behavior
func DefaultRetryConfig() RetryConfig {
	return RetryConfig{
		MaxRetries:   3,
		InitialDelay: 1 * time.Second,
		MaxDelay:     30 * time.Second,
		Multiplier:   2.0,
	}
}

// OllamaProvider implements the LLMProvider interface for local Ollama models
type OllamaProvider struct {
	baseURL     string
	model       string
	httpClient  *http.Client
	retryConfig RetryConfig
	discoverer  *discovery.Discoverer
}

// OllamaRequest represents a request to the Ollama API
type OllamaRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt,omitempty"`
	Stream bool   `json:"stream,omitempty"`
	// Format is ollama's constrained-decoding field. It is `any` rather than a
	// concrete type because the API accepts two shapes at this position: the
	// string "json", and a JSON Schema document. Populate it from
	// ModelParameters.ProviderSpecific["format"], or set it directly.
	Format  any           `json:"format,omitempty"`
	Options OllamaOptions `json:"options,omitempty"`
}

// OllamaOptions represents options for Ollama generation
type OllamaOptions struct {
	Temperature float64  `json:"temperature,omitempty"`
	TopP        float64  `json:"top_p,omitempty"`
	MaxTokens   int      `json:"num_predict,omitempty"`
	Stop        []string `json:"stop,omitempty"`
	// Seed is a POINTER so that an unset seed and a deliberate seed of 0 stay
	// distinguishable. `int` with omitempty would silently drop seed 0, which
	// is a perfectly legal seed, and the caller would get non-deterministic
	// output while believing they had pinned it.
	Seed *int `json:"seed,omitempty"`
}

// OllamaResponse represents a response from the Ollama API
type OllamaResponse struct {
	Model    string `json:"model"`
	Response string `json:"response"`
	Done     bool   `json:"done"`
	Context  []int  `json:"context,omitempty"`
	// The counters ollama reports on the final chunk. These are MEASURED by the
	// server, and they are the only real token counts available from this API —
	// everything else this adapter reports about token usage is an estimate and
	// says so. They are surfaced through LLMResponse.Metadata, and EvalCount
	// replaces the estimate in TokensUsed whenever the server sent one.
	PromptEvalCount    int   `json:"prompt_eval_count,omitempty"`
	PromptEvalDuration int64 `json:"prompt_eval_duration,omitempty"`
	EvalCount          int   `json:"eval_count,omitempty"`
	EvalDuration       int64 `json:"eval_duration,omitempty"`
	TotalDuration      int64 `json:"total_duration,omitempty"`
}

// NewOllamaProvider creates a new Ollama provider instance
func NewOllamaProvider(baseURL, model string) *OllamaProvider {
	return NewOllamaProviderWithRetry(baseURL, model, DefaultRetryConfig())
}

// NewOllamaProviderWithRetry creates a new Ollama provider instance with custom retry config
func NewOllamaProviderWithRetry(baseURL, model string, retryConfig RetryConfig) *OllamaProvider {
	if baseURL == "" {
		baseURL = settings.BaseURL("ollama", DefaultBaseURL, EnvBaseURL)
	}
	if model == "" {
		model = settings.Model("ollama", DefaultModel, EnvModel)
	}

	p := &OllamaProvider{
		baseURL: baseURL,
		model:   model,
		httpClient: &http.Client{
			Timeout: settings.Timeout("ollama", DefaultTimeout, EnvTimeout), // ollama can be slow for first requests
		},
		retryConfig: retryConfig,
	}

	p.discoverer = discovery.NewDiscoverer(discovery.ProviderConfig{
		ProviderName:   "ollama",
		ModelsEndpoint: baseURL + "/api/tags",
		ModelsDevID:    "ollama",
		APIKey:         "local",
		ResponseParser: discovery.ParseOllamaModelsResponse,
		FallbackModels: []string{
			"llama2",
			"llama2:13b",
			"llama2:70b",
			"codellama",
			"mistral",
			"vicuna",
			"orca-mini",
		},
	})

	return p
}

// SetTimeout sets the HTTP timeout for every subsequent request this provider
// makes, replacing the constructor default.
//
// WHY THIS IS NOT OPTIONAL. The default is a guess about how long a local
// model takes to answer, and it is only ever right by accident: the same
// prompt against the same ollama on a different host, or against a larger
// model, or with a longer context, can take several times as long. When the
// client timeout fires mid-decode the caller does not get a slow answer — it
// gets a transport error, and no amount of retrying at a higher layer can
// distinguish that from an ollama that is genuinely down. A caller that knows
// its own latency budget must be able to state it.
//
// A non-positive duration is rejected rather than applied, because
// http.Client.Timeout treats zero as "wait forever", which would turn a
// mistaken SetTimeout(0) into a hung request. Mirrors pkg/http.Client.SetTimeout.
func (o *OllamaProvider) SetTimeout(timeout time.Duration) {
	if timeout <= 0 {
		return
	}
	o.httpClient.Timeout = timeout
}

// Timeout reports the HTTP timeout currently in force. Present so that a
// caller — or a test — can verify the setting took, rather than assume it.
func (o *OllamaProvider) Timeout() time.Duration {
	return o.httpClient.Timeout
}

// buildRequest assembles the wire request shared by Complete and
// CompleteStream. Keeping it in one place is what stops the two paths drifting:
// before this existed, each built its own literal, so any option added to one
// was silently missing from the other.
func (o *OllamaProvider) buildRequest(req *models.LLMRequest, stream bool) OllamaRequest {
	model := o.model
	// A per-request model overrides the provider's configured one, matching how
	// the perplexity and venice adapters treat ModelParameters.Model.
	if req.ModelParams.Model != "" {
		model = req.ModelParams.Model
	}

	out := OllamaRequest{
		Model:  model,
		Prompt: req.Prompt,
		Stream: stream,
		Options: OllamaOptions{
			Temperature: req.ModelParams.Temperature,
			TopP:        req.ModelParams.TopP,
			MaxTokens:   req.ModelParams.MaxTokens,
			Stop:        req.ModelParams.StopSequences,
		},
	}

	ps := req.ModelParams.ProviderSpecific
	if ps == nil {
		return out
	}

	if v, ok := ps[ParamFormat]; ok {
		// Both shapes ollama accepts at this position, and nothing else. An
		// unrecognised type is dropped rather than forwarded, because sending a
		// `format` ollama cannot parse fails the whole generation, and failing
		// on a caller's typo is worse than ignoring it.
		switch f := v.(type) {
		case string:
			if f != "" {
				out.Format = f
			}
		case map[string]any:
			if len(f) > 0 {
				out.Format = f
			}
		case json.RawMessage:
			if len(f) > 0 {
				out.Format = f
			}
		}
	}

	if seed, ok := asInt(ps[ParamSeed]); ok {
		out.Options.Seed = &seed
	}

	// ProviderSpecific["num_predict"] is the more specific instruction, so it
	// wins over ModelParameters.MaxTokens when both are present.
	if n, ok := asInt(ps[ParamNumPredict]); ok {
		out.Options.MaxTokens = n
	}

	return out
}

// asInt accepts the numeric shapes a ProviderSpecific value can arrive in. The
// map is `map[string]interface{}`, so a value that has been through JSON is a
// float64 while the same value set in Go code is an int — accepting only one
// of those would work in a unit test and fail against a decoded request.
func asInt(v any) (int, bool) {
	switch n := v.(type) {
	case int:
		return n, true
	case int32:
		return int(n), true
	case int64:
		return int(n), true
	case float32:
		return int(n), true
	case float64:
		return int(n), true
	}
	return 0, false
}

// Complete implements the LLMProvider interface
func (o *OllamaProvider) Complete(ctx context.Context, req *models.LLMRequest) (*models.LLMResponse, error) {
	ollamaReq := o.buildRequest(req, false)

	resp, err := o.makeRequest(ctx, ollamaReq)
	if err != nil {
		return nil, fmt.Errorf("failed to complete request: %w", err)
	}

	return o.convertResponse(resp, req.ID)
}

// CompleteStream implements streaming completion
func (o *OllamaProvider) CompleteStream(ctx context.Context, req *models.LLMRequest) (<-chan *models.LLMResponse, error) {
	ch := make(chan *models.LLMResponse)

	go func() {
		defer close(ch)

		ollamaReq := o.buildRequest(req, true)

		httpReq, err := http.NewRequestWithContext(ctx, "POST", o.baseURL+"/api/generate", nil)
		if err != nil {
			ch <- &models.LLMResponse{
				RequestID:    req.ID,
				ProviderID:   "ollama",
				ProviderName: "Ollama",
				// CONST-046 round-441: error response routed through i18n.
				Content: i18n.Tr(context.Background(),
					"llmprovider_ollama_response_error",
					map[string]any{"error": err.Error()}),
				Confidence:   0.0,
				FinishReason: "error",
				CreatedAt:    time.Now(),
			}
			return
		}

		jsonData, err := json.Marshal(ollamaReq)
		if err != nil {
			ch <- &models.LLMResponse{
				RequestID:    req.ID,
				ProviderID:   "ollama",
				ProviderName: "Ollama",
				// CONST-046 round-441: error response routed through i18n.
				Content: i18n.Tr(context.Background(),
					"llmprovider_ollama_response_error",
					map[string]any{"error": err.Error()}),
				Confidence:   0.0,
				FinishReason: "error",
				CreatedAt:    time.Now(),
			}
			return
		}

		httpReq.Header.Set("Content-Type", "application/json")
		httpReq.Body = io.NopCloser(bytes.NewBuffer(jsonData))

		response, err := o.httpClient.Do(httpReq)
		if err != nil {
			ch <- &models.LLMResponse{
				RequestID:    req.ID,
				ProviderID:   "ollama",
				ProviderName: "Ollama",
				// CONST-046 round-441: error response routed through i18n.
				Content: i18n.Tr(context.Background(),
					"llmprovider_ollama_response_error",
					map[string]any{"error": err.Error()}),
				Confidence:   0.0,
				FinishReason: "error",
				CreatedAt:    time.Now(),
			}
			return
		}
		defer func() { _ = response.Body.Close() }()

		// Guard on status BEFORE decoding the body as a stream. The Ollama
		// stream endpoint returns a non-2xx status with a JSON error body
		// (e.g. 404 "model not found"). That JSON decodes cleanly into an
		// empty OllamaResponse, so without this guard the loop would emit a
		// silent empty chunk and close without ever signalling an error — a
		// success-on-HTTP-error bluff. The non-streaming Complete path already
		// returns an error for the same status; mirror that contract here.
		if response.StatusCode != http.StatusOK {
			body, _ := io.ReadAll(response.Body) //nolint:errcheck
			ch <- &models.LLMResponse{
				RequestID:    req.ID,
				ProviderID:   "ollama",
				ProviderName: "Ollama",
				Content: i18n.Tr(context.Background(),
					"llmprovider_ollama_response_error",
					map[string]any{"error": fmt.Sprintf(
						"Ollama API returned status %d: %s",
						response.StatusCode, string(body))}),
				Confidence:   0.0,
				FinishReason: "error",
				CreatedAt:    time.Now(),
			}
			return
		}

		decoder := json.NewDecoder(response.Body)
		fullContent := ""

		for {
			var streamResp OllamaResponse
			if err := decoder.Decode(&streamResp); err != nil {
				if err == io.EOF {
					break
				}
				// Send error response and exit on decode error
				select {
				case ch <- &models.LLMResponse{
					RequestID:    req.ID,
					ProviderID:   "ollama",
					ProviderName: "Ollama",
					// CONST-046 round-441: decode-error response routed through i18n.
					Content: i18n.Tr(context.Background(),
						"llmprovider_ollama_response_decode_error",
						map[string]any{"error": err.Error()}),
					Confidence:   0.0,
					FinishReason: "error",
					CreatedAt:    time.Now(),
				}:
				case <-ctx.Done():
				}
				return
			}

			fullContent += streamResp.Response

			chunkResp := &models.LLMResponse{
				RequestID:      req.ID,
				ProviderID:     "ollama",
				ProviderName:   "Ollama",
				Content:        streamResp.Response,
				Confidence:     0.8,
				TokensUsed:     1,
				ResponseTime:   time.Now().UnixMilli(),
				FinishReason:   "",
				Selected:       false,
				SelectionScore: 0.0,
				CreatedAt:      time.Now(),
			}

			select {
			case ch <- chunkResp:
			case <-ctx.Done():
				return
			}

			if streamResp.Done {
				// Send final response. The final chunk is where ollama reports
				// its measured counters, so this is the only place they can be
				// read on the streaming path — and the only place a real token
				// count is available at all.
				tokens := len(fullContent) / 4
				estimated := true
				if streamResp.EvalCount > 0 {
					tokens = streamResp.EvalCount
					estimated = false
				}
				meta := countersOf(&streamResp)
				meta["tokens_estimated"] = estimated

				finalResp := &models.LLMResponse{
					RequestID:      req.ID,
					ProviderID:     "ollama",
					ProviderName:   "Ollama",
					Content:        "",
					Confidence:     0.8,
					TokensUsed:     tokens,
					ResponseTime:   time.Now().UnixMilli(),
					FinishReason:   "stop",
					Metadata:       meta,
					Selected:       false,
					SelectionScore: 0.0,
					CreatedAt:      time.Now(),
				}
				ch <- finalResp
				break
			}
		}
	}()

	return ch, nil
}

// HealthCheck implements health checking for the Ollama provider
func (o *OllamaProvider) HealthCheck() error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, "GET", o.baseURL+"/api/tags", nil)
	if err != nil {
		return fmt.Errorf("failed to create health check request: %w", err)
	}

	resp, err := o.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("health check request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("health check failed with status: %d", resp.StatusCode)
	}

	return nil
}

// GetCapabilities returns the capabilities of the Ollama provider
func (o *OllamaProvider) GetCapabilities() *models.ProviderCapabilities {
	return &models.ProviderCapabilities{
		SupportedModels: o.discoverer.DiscoverModels(),
		SupportedFeatures: []string{
			"text_completion",
			"chat",
			"streaming",
		},
		SupportedRequestTypes: []string{
			"text_completion",
			"chat",
		},
		SupportsStreaming:       true,
		SupportsFunctionCalling: false,
		SupportsVision:          false,
		Limits: models.ModelLimits{
			MaxTokens:             4096,
			MaxInputLength:        4096,
			MaxOutputLength:       4096,
			MaxConcurrentRequests: 1, // Ollama typically handles one request at a time
		},
		Metadata: map[string]string{
			"provider":     "Ollama",
			"model_family": "Local Models",
			"api_version":  "v1",
			"local":        "true",
		},
	}
}

// ValidateConfig validates the provider configuration
func (o *OllamaProvider) ValidateConfig(config map[string]interface{}) (bool, []string) {
	var errors []string

	if o.baseURL == "" {
		// CONST-046 round-425: validation error routed through i18n.
		errors = append(errors, i18n.Tr(context.Background(),
			"llmprovider_validate_base_url_required", nil))
	}

	if o.model == "" {
		// CONST-046 round-425: validation error routed through i18n.
		errors = append(errors, i18n.Tr(context.Background(),
			"llmprovider_validate_model_required", nil))
	}

	return len(errors) == 0, errors
}

// countersOf exposes ollama's server-measured counters through Metadata, under
// the API's own key names so a caller does not have to learn a second
// vocabulary. Only counters the server actually sent are included: a zero here
// would be indistinguishable from "ollama reported zero", and inventing one
// would make an absent measurement look like a measured absence.
func countersOf(resp *OllamaResponse) map[string]interface{} {
	m := map[string]interface{}{
		"model":   resp.Model,
		"context": len(resp.Context),
	}
	if resp.PromptEvalCount != 0 {
		m["prompt_eval_count"] = resp.PromptEvalCount
	}
	if resp.PromptEvalDuration != 0 {
		m["prompt_eval_duration"] = resp.PromptEvalDuration
	}
	if resp.EvalCount != 0 {
		m["eval_count"] = resp.EvalCount
	}
	if resp.EvalDuration != 0 {
		m["eval_duration"] = resp.EvalDuration
	}
	if resp.TotalDuration != 0 {
		m["total_duration"] = resp.TotalDuration
	}
	return m
}

// convertResponse converts Ollama API response to internal format
func (o *OllamaProvider) convertResponse(resp *OllamaResponse, requestID string) (*models.LLMResponse, error) {
	// EvalCount is ollama's own count of generated tokens. When the server sent
	// one, use it and label nothing an estimate; otherwise fall back to the
	// length heuristic, which IS an estimate and is recorded as such.
	tokens := len(resp.Response) / 4
	estimated := true
	if resp.EvalCount > 0 {
		tokens = resp.EvalCount
		estimated = false
	}

	meta := countersOf(resp)
	meta["tokens_estimated"] = estimated

	return &models.LLMResponse{
		ID:             fmt.Sprintf("ollama-%d", time.Now().Unix()),
		RequestID:      requestID,
		ProviderID:     "ollama",
		ProviderName:   "Ollama",
		Content:        resp.Response,
		Confidence:     0.8, // Ollama doesn't provide confidence scores
		TokensUsed:     tokens,
		ResponseTime:   time.Now().UnixMilli(),
		FinishReason:   "stop",
		Metadata:       meta,
		Selected:       false,
		SelectionScore: 0.0,
		CreatedAt:      time.Now(),
	}, nil
}

// makeRequest sends a request to the Ollama API with retry logic
func (o *OllamaProvider) makeRequest(ctx context.Context, req OllamaRequest) (*OllamaResponse, error) {
	jsonData, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	var lastErr error
	delay := o.retryConfig.InitialDelay

	for attempt := 0; attempt <= o.retryConfig.MaxRetries; attempt++ {
		// Check context before making request
		select {
		case <-ctx.Done():
			return nil, fmt.Errorf("context cancelled: %w", ctx.Err())
		default:
		}

		httpReq, err := http.NewRequestWithContext(ctx, "POST", o.baseURL+"/api/generate", bytes.NewBuffer(jsonData))
		if err != nil {
			return nil, fmt.Errorf("failed to create HTTP request: %w", err)
		}

		httpReq.Header.Set("Content-Type", "application/json")

		resp, err := o.httpClient.Do(httpReq)
		if err != nil {
			lastErr = fmt.Errorf("HTTP request failed: %w", err)
			if attempt < o.retryConfig.MaxRetries {
				o.waitWithJitter(ctx, delay)
				delay = o.nextDelay(delay)
				continue
			}
			return nil, lastErr
		}

		// Check for retryable status codes
		if isRetryableStatus(resp.StatusCode) && attempt < o.retryConfig.MaxRetries {
			_ = resp.Body.Close()
			lastErr = fmt.Errorf("HTTP %d: retryable error", resp.StatusCode)
			o.waitWithJitter(ctx, delay)
			delay = o.nextDelay(delay)
			continue
		}

		body, err := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		if err != nil {
			return nil, fmt.Errorf("failed to read response body: %w", err)
		}

		if resp.StatusCode != http.StatusOK {
			return nil, fmt.Errorf("Ollama API returned status %d: %s", resp.StatusCode, string(body))
		}

		var ollamaResp OllamaResponse
		if err := json.Unmarshal(body, &ollamaResp); err != nil {
			return nil, fmt.Errorf("failed to unmarshal response: %w", err)
		}

		return &ollamaResp, nil
	}

	return nil, fmt.Errorf("all %d retry attempts failed: %w", o.retryConfig.MaxRetries+1, lastErr)
}

// isRetryableStatus returns true for HTTP status codes that warrant a retry
func isRetryableStatus(statusCode int) bool {
	switch statusCode {
	case http.StatusTooManyRequests, // 429 - Rate limited
		http.StatusInternalServerError, // 500
		http.StatusBadGateway,          // 502
		http.StatusServiceUnavailable,  // 503
		http.StatusGatewayTimeout:      // 504
		return true
	default:
		return false
	}
}

// waitWithJitter waits for the specified duration plus random jitter
func (o *OllamaProvider) waitWithJitter(ctx context.Context, delay time.Duration) {
	// Add 10% jitter - using math/rand is acceptable for non-security jitter
	jitter := time.Duration(rand.Float64() * 0.1 * float64(delay)) // #nosec G404 - jitter doesn't require cryptographic randomness
	select {
	case <-ctx.Done():
	case <-time.After(delay + jitter):
	}
}

// nextDelay calculates the next delay using exponential backoff
func (o *OllamaProvider) nextDelay(currentDelay time.Duration) time.Duration {
	nextDelay := time.Duration(float64(currentDelay) * o.retryConfig.Multiplier)
	if nextDelay > o.retryConfig.MaxDelay {
		nextDelay = o.retryConfig.MaxDelay
	}
	return nextDelay
}
