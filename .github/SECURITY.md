# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| latest  | Yes                |

## Reporting a Vulnerability

If you discover a security vulnerability in QuantPath, please report it
responsibly:

1. **Do NOT open a public GitHub issue.**
2. Email security concerns to the maintainers via a private GitHub
   Security Advisory:
   https://github.com/MasterAgentAI/QuantPath/security/advisories/new
3. Include a description of the vulnerability and steps to reproduce.
4. We will acknowledge receipt within 48 hours and provide a timeline
   for a fix.

## Scope

This policy covers:
- The QuantPath CLI and web application code
- Data processing pipelines (scraping, parsing, training)
- The Streamlit web dashboard

This policy does NOT cover:
- Third-party services (GradCafe, QuantNet, LinkedIn)
- User-provided profile YAML files
- The Anthropic API integration (report to Anthropic directly)

## Data Privacy

QuantPath processes user-provided academic profiles locally. No personal
data is transmitted to external services unless the user explicitly runs
the AI advisor tool (which sends profile data to the Anthropic API).

Admission data in this repository has been aggregated from public sources
and does not contain personally identifiable information (PII).
