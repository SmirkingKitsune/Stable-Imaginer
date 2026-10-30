"""Pairing codes, proofs and tokens.

The code is shown on one instance and typed into the other; it never crosses the network.
The requester proves it knows the code with an HMAC over a fresh challenge from the
responder and both instance ids, so a proof can't be replayed or used elsewhere. Each side
then issues the other a random token, and keeps only the hash of the one it issued.
"""

import hashlib
import hmac
import re
import secrets

# Crockford's base32: no I, L, O or U, so a code read aloud or retyped survives.
ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
CODE_LENGTH = 8  # 40 bits, and a code dies after MAX_FAILED_ATTEMPTS wrong tries
CODE_TTL_S = 600.0
CHALLENGE_TTL_S = 120.0
MAX_FAILED_ATTEMPTS = 5
_SEPARATORS = re.compile(r"[\s-]+")


def new_code() -> str:
    """A fresh code, shown as ``XXXX-XXXX``."""
    raw = "".join(secrets.choice(ALPHABET) for _ in range(CODE_LENGTH))
    return f"{raw[:4]}-{raw[4:]}"


def normalize_code(code: str) -> str:
    """What the user typed, the way it was minted: case and separators don't matter, and
    the letters Crockford's alphabet leaves out read as the digits they look like."""
    cleaned = _SEPARATORS.sub("", code).upper()
    return cleaned.translate(str.maketrans("ILO", "110"))


def code_key(code: str) -> str:
    """The stored form of a code, and the HMAC key of a proof."""
    return hashlib.sha256(normalize_code(code).encode()).hexdigest()


def new_challenge() -> str:
    return secrets.token_urlsafe(24)


def proof(key: str, challenge: str, requester_id: str, responder_id: str) -> str:
    message = "\n".join(("imaginer-pair/1", challenge, requester_id, responder_id)).encode()
    return hmac.new(bytes.fromhex(key), message, hashlib.sha256).hexdigest()


def identity_proof(token_sha256: str, nonce: str, prover_id: str) -> str:
    """Proof that the prover is the linked instance, without sending a token: keyed with the
    hash of the token we call it with, which only it (and we) know."""
    message = "\n".join(("imaginer-link/1", nonce, prover_id)).encode()
    return hmac.new(bytes.fromhex(token_sha256), message, hashlib.sha256).hexdigest()


def proof_matches(expected: str, presented: str) -> bool:
    return hmac.compare_digest(expected.encode(), presented.lower().encode())


def new_token() -> str:
    return secrets.token_urlsafe(32)


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()
