# Security Policy

ProVe is research code that serves language models over HTTP through a fork of SGLang. Run it on
trusted networks only; the server has no authentication of its own.

## Reporting a vulnerability

Please do not open a public issue for security problems. Report them by e-mail to the corresponding
authors listed in the paper (weiwen.xu@bytedance.com) with a description of the issue and, if possible,
steps to reproduce. We will acknowledge the report within a few working days and coordinate a fix and
disclosure with you.

## Scope

* Code in this repository (`inference/`, `train/`, `eval/`).
* Vulnerabilities in the bundled SGLang fork should also be reported upstream to the SGLang project
  when they affect unmodified upstream code.
