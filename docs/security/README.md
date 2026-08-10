# Security and trust boundaries

Active Fedference is research software, not a deployment-ready secure
federation service. Start with the
[repository threat model](active_fedference-threat-model.md) before extending
the transport, exposing a listener, or making authentication, privacy, or
Byzantine-robustness claims.

The current network adapter is deliberately loopback-only. Its HMAC mode
provides shared-key frame integrity; it does not identify individual workers,
encrypt beliefs, provide forward secrecy, or tolerate a compromised worker.
The planned Docker emulator and physical multi-host lane have separate
acceptance criteria in
[the multi-machine roadmap](../todo/true-multi-machine-federation.md).
