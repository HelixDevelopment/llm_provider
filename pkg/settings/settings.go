// Package settings is the central authority within this module for resolving
// per-provider NON-CREDENTIAL configuration from the environment: which model a
// provider defaults to, which endpoint it talks to, and how long it waits.
//
// WHY IT EXISTS. Every adapter in this module carried its operational defaults
// as frozen literals — `if model == "" { model = CerebrasModel }`, with
// `CerebrasModel = "llama3.1-8b"` a package constant and nothing behind it. A
// default is a reasonable starting point; a default with no override layer is
// a decision the library has taken away from the operator. The failure that
// makes this concrete: a vendor retires or rate-caps a model, and every
// consumer that passed an empty model string starts silently talking to a
// model that no longer exists — a code change, a release and a dependency bump
// away from being fixable, when it should have been an environment variable.
// The same argument applies to a frozen `http://localhost:11434` on a host that
// runs its inference elsewhere.
//
// This package is deliberately the SIBLING of pkg/apikeys, not a copy of it.
// apikeys resolves credentials and is the sole reader of `ApiKey_*`; this one
// resolves everything that is NOT a credential and is the sole reader of
// `LLMPROVIDER_*`. Keeping them apart matters: a credential must never be
// logged or echoed, and the values here routinely are — an endpoint appears in
// error messages, a model name in metadata. Merging the two would put a secret
// one careless fmt.Printf away from a log file.
//
// CONVENTION
//
//	LLMPROVIDER_<PROVIDER>_MODEL      the default model
//	LLMPROVIDER_<PROVIDER>_BASE_URL   the default endpoint
//	LLMPROVIDER_<PROVIDER>_TIMEOUT    the default HTTP timeout (time.ParseDuration)
//
// <PROVIDER> is the adapter's package name upper-cased, with every character
// that is not a letter or digit folded to `_` — so `zen_http` and `Zen HTTP`
// both become `ZEN_HTTP`.
//
// ALIASES. Several backends already have a well-known variable of their own
// that an operator is likely to have exported for the vendor's CLI — OLLAMA_HOST
// is the obvious one. Each resolver accepts those as trailing alias names and
// consults them AFTER the canonical key, so this module's own convention always
// wins where both are set, and an operator who has configured the vendor's tool
// gets this library pointed at the same place for free.
//
// PRECEDENCE, highest first:
//
//  1. an explicit argument passed by the caller (each adapter checks this
//     before calling in here at all)
//  2. LLMPROVIDER_<PROVIDER>_<SETTING>
//  3. the aliases, in the order given
//  4. the compiled-in fallback
//
// An env var that is set but EMPTY counts as unset. That is what lets
// `LLMPROVIDER_OLLAMA_MODEL= go test ./...` restore the compiled default
// without unsetting anything, and it is what makes a test hermetic.
package settings

import (
	"os"
	"strings"
	"time"
)

// Prefix is the canonical env-var prefix for this module's non-credential
// settings. It is deliberately distinct from apikeys.Prefix.
const Prefix = "LLMPROVIDER_"

// Setting name suffixes.
const (
	SuffixModel   = "MODEL"
	SuffixBaseURL = "BASE_URL"
	SuffixTimeout = "TIMEOUT"
)

// Key returns the canonical environment-variable name for one provider setting,
// e.g. Key("ollama", SuffixModel) == "LLMPROVIDER_OLLAMA_MODEL". Exported so a
// test, a doc generator or an operator-facing diagnostic can print the exact
// name to set rather than reconstructing the convention by hand and getting it
// subtly wrong.
func Key(provider, suffix string) string {
	return Prefix + normalize(provider) + "_" + suffix
}

// normalize folds a provider name into the env-var alphabet: upper case, with
// every character that is neither a letter nor a digit replaced by '_'.
func normalize(provider string) string {
	var b strings.Builder
	b.Grow(len(provider))
	for _, r := range provider {
		switch {
		case r >= 'a' && r <= 'z':
			b.WriteRune(r - 32)
		case (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9'):
			b.WriteRune(r)
		default:
			b.WriteByte('_')
		}
	}
	return b.String()
}

// lookup returns the first non-empty value among the canonical key and the
// aliases, plus the name it came from. The name is returned so callers that
// want to report WHERE a setting came from can do so; a value with no
// provenance is hard to debug.
func lookup(provider, suffix string, aliases []string) (value, from string, ok bool) {
	names := append([]string{Key(provider, suffix)}, aliases...)
	for _, n := range names {
		if v := os.Getenv(n); v != "" {
			return v, n, true
		}
	}
	return "", "", false
}

// String resolves an arbitrary string setting for a provider.
func String(provider, suffix, fallback string, aliases ...string) string {
	if v, _, ok := lookup(provider, suffix, aliases); ok {
		return v
	}
	return fallback
}

// Model resolves the default model for a provider.
func Model(provider, fallback string, aliases ...string) string {
	return String(provider, SuffixModel, fallback, aliases...)
}

// BaseURL resolves the default endpoint for a provider.
func BaseURL(provider, fallback string, aliases ...string) string {
	return String(provider, SuffixBaseURL, fallback, aliases...)
}

// Timeout resolves the default HTTP timeout for a provider. The value is parsed
// with time.ParseDuration, so "180s", "3m" and "1h30m" all work.
//
// A value that does not parse, or that is not positive, is IGNORED and the
// fallback is used. It is not treated as zero, because http.Client.Timeout
// reads zero as "wait forever" — so honouring a mistyped `LLMPROVIDER_X_TIMEOUT`
// literally would turn an operator's attempt to SHORTEN a timeout into an
// unbounded request. Falling back is the conservative reading of the intent.
func Timeout(provider string, fallback time.Duration, aliases ...string) time.Duration {
	raw, _, ok := lookup(provider, SuffixTimeout, aliases)
	if !ok {
		return fallback
	}
	d, err := time.ParseDuration(raw)
	if err != nil || d <= 0 {
		return fallback
	}
	return d
}

// Source reports the environment-variable name a setting would be read from,
// or "" when none is set and the compiled fallback would win. Intended for
// diagnostics: "which of my three variables is actually in force?" is a
// question an operator should not have to answer by bisection.
func Source(provider, suffix string, aliases ...string) string {
	_, from, _ := lookup(provider, suffix, aliases)
	return from
}
