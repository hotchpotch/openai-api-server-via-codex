package app

import "regexp"

var sensitivePatterns = []*regexp.Regexp{
	regexp.MustCompile(`(?i)(\bbearer\s+)[^\s,;"']+`),
	regexp.MustCompile(`(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,;"']+`),
	regexp.MustCompile(`(?i)((?:access_token|refresh_token|id_token|api_key)\s*[=:]\s*)[^\s,;"']+`),
	regexp.MustCompile(`\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b`),
}
var logControlCharacters = regexp.MustCompile(`[\x00-\x1f\x7f]`)

func redactSensitive(value string) string {
	for _, pattern := range sensitivePatterns {
		value = pattern.ReplaceAllString(value, `${1}[REDACTED]`)
	}
	return logControlCharacters.ReplaceAllString(value, " ")
}
