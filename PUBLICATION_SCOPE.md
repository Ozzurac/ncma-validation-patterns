# Publication boundary

## Source policy

This public code is an **independently written, simplified adaptation of engineering principles** observed in private NCMA work. It is not an export or a historical branch of the original code. There is no path, import, interface or network dependency on a private repo.

## What is intentionally excluded

- Uncommitted or unpushed workspace changes, OCI/Qwen experiments and private Git history.
- Original Jarvis planner, model router, memory, execution policy, desktop adapters and fine-tuning work.
- Original ModForge compiler, parser, resolver, game corpus and semantic rules.
- Gateway executor, private API routes, tunnel/security policy and deployment configuration.
- Secrets, environment files, credentials, personal media, family conversations, internal hostnames and local paths.
- Employer/client data, support incidents, logs, proprietary content and game assets.
- Historical validation artifacts that have not been independently reviewed for redistribution.

## Safe-publication checklist

1. Create independent sources with fictional data in a fresh staging directory.
2. Run unit tests, negative tests, sample demo and compilation.
3. Inspect the **exact Git index** for unexpected paths and build artifacts.
4. Scan tracked content for secret patterns, private hostnames and filesystem paths.
5. Check whitespace, metadata, license, dependencies and provenance.
6. Create an isolated public repository rather than changing a private repo's visibility.
7. Confirm the remote file list and GitHub Actions outcome.

**Important:** `.gitignore` never erases secrets already present in Git history. The files in this repo are generated separately, without preserving private history.

Static scanning helps but is not an absolute guarantee. Future contributions must receive the same review.