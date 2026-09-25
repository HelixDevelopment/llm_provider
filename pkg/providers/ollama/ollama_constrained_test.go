package ollama

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"digital.vasic.llmprovider/pkg/models"
)

// The variables an operator actually exports, written out in full rather than
// recomputed with settings.Key(): a test that derives the key from the same
// helper its subject uses agrees with that subject by construction and stays
// green even when the key is wrong. The convention is pinned independently by
// pkg/settings/keys_contract_test.go.
const (
	envBaseURLKey = "LLMPROVIDER_OLLAMA_BASE_URL"
	envModelKey   = "LLMPROVIDER_OLLAMA_MODEL"
	envTimeoutKey = "LLMPROVIDER_OLLAMA_TIMEOUT"
)

// These tests gate the constrained-decoding, seeding and timeout capabilities.
//
// They assert on the JSON that actually LEAVES the process, decoded into a
// map, rather than on the Go struct that produced it. That distinction is the
// whole point: a `Format` field with a wrong struct tag, or one dropped by
// `omitempty`, still populates the struct perfectly and still sends nothing.
// Only the wire body proves the option reached ollama.

// capture spins up a server that records the raw request body and answers with
// the supplied response. It returns the provider pointed at that server and a
// pointer to the decoded body, populated once a request has been made.
func capture(t *testing.T, reply OllamaResponse) (*OllamaProvider, *map[string]any) {
	t.Helper()
	body := map[string]any{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.NoError(t, json.NewDecoder(r.Body).Decode(&body))
		w.Header().Set("Content-Type", "application/json")
		require.NoError(t, json.NewEncoder(w).Encode(reply))
	}))
	t.Cleanup(srv.Close)
	return NewOllamaProvider(srv.URL, "llama2"), &body
}

func req(ps map[string]interface{}) *models.LLMRequest {
	return &models.LLMRequest{
		ID:          "req-1",
		Prompt:      "test prompt",
		ModelParams: models.ModelParameters{Temperature: 0, ProviderSpecific: ps},
	}
}

// --- format ---------------------------------------------------------------

func TestComplete_SendsJSONSchemaFormatOverTheWire(t *testing.T) {
	schema := map[string]any{
		"type":     "object",
		"required": []any{"answer"},
		"properties": map[string]any{
			"answer": map[string]any{"type": "string"},
		},
	}
	p, body := capture(t, OllamaResponse{Model: "llama2", Response: `{"answer":"x"}`, Done: true})

	_, err := p.Complete(context.Background(), req(map[string]interface{}{ParamFormat: schema}))
	require.NoError(t, err)

	got, ok := (*body)["format"]
	require.True(t, ok, "the request body carries no 'format' key at all — "+
		"constrained decoding was requested and silently dropped")
	gotMap, ok := got.(map[string]any)
	require.True(t, ok, "'format' is %T, want a JSON object", got)
	assert.Equal(t, "object", gotMap["type"])
	assert.Contains(t, gotMap, "properties")
}

func TestComplete_SendsStringFormat(t *testing.T) {
	p, body := capture(t, OllamaResponse{Model: "llama2", Response: "{}", Done: true})
	_, err := p.Complete(context.Background(), req(map[string]interface{}{ParamFormat: "json"}))
	require.NoError(t, err)
	assert.Equal(t, "json", (*body)["format"])
}

func TestComplete_OmitsFormatWhenNotRequested(t *testing.T) {
	p, body := capture(t, OllamaResponse{Model: "llama2", Response: "hi", Done: true})
	_, err := p.Complete(context.Background(), req(nil))
	require.NoError(t, err)
	assert.NotContains(t, *body, "format",
		"an unrequested 'format' must not appear — an empty one is not the same as none")
}

func TestComplete_DropsUnusableFormatRatherThanFailingTheGeneration(t *testing.T) {
	p, body := capture(t, OllamaResponse{Model: "llama2", Response: "hi", Done: true})
	_, err := p.Complete(context.Background(),
		req(map[string]interface{}{ParamFormat: 42}))
	require.NoError(t, err)
	assert.NotContains(t, *body, "format")
}

func TestCompleteStream_SendsFormat(t *testing.T) {
	schema := map[string]any{"type": "object"}
	body := map[string]any{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.NoError(t, json.NewDecoder(r.Body).Decode(&body))
		w.Header().Set("Content-Type", "application/json")
		enc := json.NewEncoder(w)
		require.NoError(t, enc.Encode(OllamaResponse{Model: "llama2", Response: "{}", Done: true, EvalCount: 7}))
	}))
	defer srv.Close()

	p := NewOllamaProvider(srv.URL, "llama2")
	ch, err := p.CompleteStream(context.Background(), req(map[string]interface{}{ParamFormat: schema}))
	require.NoError(t, err)
	for range ch { //nolint:revive // drain
	}
	assert.Contains(t, body, "format",
		"the streaming path built its own request literal and dropped 'format'")
}

// --- seed -----------------------------------------------------------------

func TestComplete_SeedZeroIsSentNotDropped(t *testing.T) {
	// The regression this guards: `Seed int` with omitempty drops seed 0, a
	// legal seed, and the caller gets non-deterministic output while believing
	// the run was pinned.
	p, body := capture(t, OllamaResponse{Model: "llama2", Response: "hi", Done: true})
	_, err := p.Complete(context.Background(), req(map[string]interface{}{ParamSeed: 0}))
	require.NoError(t, err)

	opts, ok := (*body)["options"].(map[string]any)
	require.True(t, ok, "no options object in the request body")
	require.Contains(t, opts, "seed", "seed 0 was dropped")
	assert.Equal(t, float64(0), opts["seed"])
}

func TestComplete_SeedAcceptsJSONDecodedNumber(t *testing.T) {
	// ProviderSpecific is map[string]interface{}; a value that has been through
	// JSON arrives as float64, not int.
	p, body := capture(t, OllamaResponse{Model: "llama2", Response: "hi", Done: true})
	_, err := p.Complete(context.Background(), req(map[string]interface{}{ParamSeed: float64(1337)}))
	require.NoError(t, err)
	opts := (*body)["options"].(map[string]any)
	assert.Equal(t, float64(1337), opts["seed"])
}

func TestComplete_OmitsSeedWhenNotRequested(t *testing.T) {
	p, body := capture(t, OllamaResponse{Model: "llama2", Response: "hi", Done: true})
	_, err := p.Complete(context.Background(), req(nil))
	require.NoError(t, err)
	opts, ok := (*body)["options"].(map[string]any)
	if ok {
		assert.NotContains(t, opts, "seed")
	}
}

// --- num_predict and model override ---------------------------------------

func TestComplete_NumPredictOverridesMaxTokens(t *testing.T) {
	p, body := capture(t, OllamaResponse{Model: "llama2", Response: "hi", Done: true})
	r := req(map[string]interface{}{ParamNumPredict: 64})
	r.ModelParams.MaxTokens = 8
	_, err := p.Complete(context.Background(), r)
	require.NoError(t, err)
	opts := (*body)["options"].(map[string]any)
	assert.Equal(t, float64(64), opts["num_predict"],
		"the provider-specific key must win over MaxTokens")
}

func TestComplete_PerRequestModelOverridesProviderModel(t *testing.T) {
	p, body := capture(t, OllamaResponse{Model: "other", Response: "hi", Done: true})
	r := req(nil)
	// A synthetic id, not a real one: what is under test is that the
	// per-request field reaches the wire, and naming a real model here would
	// freeze a model choice into a backend-agnostic adapter without adding a
	// single assertion. Matches the "explicit-model" fixture below.
	r.ModelParams.Model = "per-request-model"
	_, err := p.Complete(context.Background(), r)
	require.NoError(t, err)
	assert.Equal(t, "per-request-model", (*body)["model"])
}

// --- timeout --------------------------------------------------------------

// Neither of the two tests below issues a request: they read back the timeout
// field only. The base URL is therefore inert, and it is spelled with an
// RFC 2606 .invalid host rather than a loopback address and port so that it
// cannot be mistaken for a service this adapter expects to find running, and
// so that a future edit which DOES make a request here fails loudly instead of
// reaching whatever happens to be listening on the developer's machine.
const unusedBaseURL = "http://fixture.invalid"

func TestSetTimeout(t *testing.T) {
	p := NewOllamaProvider(unusedBaseURL, "llama2")
	assert.Equal(t, DefaultTimeout, p.Timeout(), "constructor default")

	p.SetTimeout(300 * time.Second)
	assert.Equal(t, 300*time.Second, p.Timeout(), "SetTimeout did not take")
}

func TestSetTimeout_RejectsNonPositive(t *testing.T) {
	// http.Client.Timeout == 0 means "wait forever", so a mistaken
	// SetTimeout(0) must not turn a bounded request into a hung one.
	p := NewOllamaProvider(unusedBaseURL, "llama2")
	p.SetTimeout(45 * time.Second)
	p.SetTimeout(0)
	assert.Equal(t, 45*time.Second, p.Timeout())
	p.SetTimeout(-1 * time.Second)
	assert.Equal(t, 45*time.Second, p.Timeout())
}

func TestSetTimeout_ActuallyBoundsARequest(t *testing.T) {
	// Proves the setting reaches the transport, not just the field. A server
	// that never answers must produce a transport error inside the budget.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(2 * time.Second)
	}))
	defer srv.Close()

	p := NewOllamaProviderWithRetry(srv.URL, "llama2", RetryConfig{MaxRetries: 0})
	p.SetTimeout(150 * time.Millisecond)

	started := time.Now()
	_, err := p.Complete(context.Background(), req(nil))
	elapsed := time.Since(started)

	require.Error(t, err)
	assert.Less(t, elapsed, 1500*time.Millisecond,
		"the request was not bounded by the timeout that was set")
}

// --- environment layer ----------------------------------------------------

// clearCanonical empties this module's own LLMPROVIDER_OLLAMA_* keys for the
// duration of a test. Without it these tests would assert about the alias layer
// while an operator's canonical key silently outranked it — the test would pass
// on a clean workstation and fail on a configured one, which is the ambient-
// condition defect the settings package exists to make visible.
func clearCanonical(t *testing.T) {
	t.Helper()
	t.Setenv(envBaseURLKey, "")
	t.Setenv(envModelKey, "")
	t.Setenv(envTimeoutKey, "")
}

func TestConstructorReadsEnvironment(t *testing.T) {
	clearCanonical(t)
	t.Setenv(EnvBaseURL, "http://ollama.internal:9999")
	t.Setenv(EnvModel, "env-chosen-model")
	t.Setenv(EnvTimeout, "7m")

	p := NewOllamaProvider("", "")
	assert.Equal(t, "http://ollama.internal:9999", p.baseURL)
	assert.Equal(t, "env-chosen-model", p.model)
	assert.Equal(t, 7*time.Minute, p.Timeout())
}

func TestExplicitArgumentsBeatTheEnvironment(t *testing.T) {
	clearCanonical(t)
	t.Setenv(EnvBaseURL, "http://ollama.internal:9999")
	t.Setenv(EnvModel, "env-chosen-model")

	p := NewOllamaProvider("http://explicit:1234", "explicit-model")
	assert.Equal(t, "http://explicit:1234", p.baseURL)
	assert.Equal(t, "explicit-model", p.model)
}

func TestUnparseableTimeoutFallsBackRatherThanMeaningForever(t *testing.T) {
	clearCanonical(t)
	t.Setenv(EnvTimeout, "two hundred seconds")
	assert.Equal(t, DefaultTimeout, NewOllamaProvider("", "").Timeout())

	t.Setenv(EnvTimeout, "0s")
	assert.Equal(t, DefaultTimeout, NewOllamaProvider("", "").Timeout())
}

func TestConstructorDefaultsWhenEnvironmentIsClear(t *testing.T) {
	clearCanonical(t)
	t.Setenv(EnvBaseURL, "")
	t.Setenv(EnvModel, "")
	t.Setenv(EnvTimeout, "")

	p := NewOllamaProvider("", "")
	assert.Equal(t, DefaultBaseURL, p.baseURL)
	assert.Equal(t, DefaultModel, p.model)
	assert.Equal(t, DefaultTimeout, p.Timeout())
}

// --- server-measured counters ---------------------------------------------

func TestComplete_PrefersServerTokenCountOverTheEstimate(t *testing.T) {
	p, _ := capture(t, OllamaResponse{
		Model: "llama2", Response: "a much longer response than four characters", Done: true,
		PromptEvalCount: 11, PromptEvalDuration: 1000, EvalCount: 23, EvalDuration: 2000, TotalDuration: 3000,
	})
	resp, err := p.Complete(context.Background(), req(nil))
	require.NoError(t, err)

	assert.Equal(t, 23, resp.TokensUsed, "the server's own eval_count must win over len/4")
	assert.Equal(t, false, resp.Metadata["tokens_estimated"])
	assert.Equal(t, 11, resp.Metadata["prompt_eval_count"])
	assert.Equal(t, int64(3000), resp.Metadata["total_duration"])
}

func TestComplete_LabelsTheEstimateWhenTheServerSentNoCounters(t *testing.T) {
	p, _ := capture(t, OllamaResponse{Model: "llama2", Response: "abcdefgh", Done: true})
	resp, err := p.Complete(context.Background(), req(nil))
	require.NoError(t, err)

	assert.Equal(t, 2, resp.TokensUsed)
	assert.Equal(t, true, resp.Metadata["tokens_estimated"])
	assert.NotContains(t, resp.Metadata, "eval_count",
		"an absent counter must stay absent, not be reported as zero")
}
