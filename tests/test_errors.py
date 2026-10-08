"""The public response is an allowlist, never a projection of raw exceptions."""

import json
import unittest

from validation_patterns.errors import PublicError, parse_known_error


class ErrorTransportTests(unittest.TestCase):
    def test_flat_recognized_error(self):
        error = parse_known_error(409, {"error": "STATE_CONFLICT", "detail": "internal"})
        self.assertIsInstance(error, PublicError)
        self.assertEqual(error.code, "STATE_CONFLICT")
        self.assertEqual(error.http_status, 409)
        self.assertFalse(error.retryable)

    def test_nested_transient_error(self):
        error = parse_known_error(
            503, {"error": {"code": "SERVICE_UNAVAILABLE", "retryable": True}}
        )
        self.assertTrue(error.retryable)
        self.assertEqual(error.to_dict()["status"], 503)

    def test_upstream_secrets_are_not_redistributed(self):
        sentinel = "do-not-expose-this-private-token"
        payload = {
            "error": {
                "code": "STATE_CONFLICT",
                "detail": sentinel,
                "authorization": f"Bearer {sentinel}",
                "traceback": f"C:/private/{sentinel}",
                "details": {"credentials": sentinel},
                "retryable": True,
            },
            "headers": {"Authorization": sentinel},
        }
        result = parse_known_error(409, payload)
        emitted = json.dumps(result.to_dict())
        self.assertNotIn(sentinel, emitted)
        self.assertNotIn("authorization", emitted.lower())
        self.assertNotIn("traceback", emitted.lower())
        self.assertNotIn("details", emitted.lower())

    def test_unrecognized_code_is_not_reclassified(self):
        self.assertIsNone(parse_known_error(409, {"error": "UNKNOWN_VENDOR_ERROR"}))

    def test_non_json_like_response_is_unknown(self):
        self.assertIsNone(parse_known_error(502, "Bad Gateway"))

    def test_no_error_field_is_unknown(self):
        self.assertIsNone(parse_known_error(500, {"detail": "internal"}))

    def test_error_objects_must_contain_code(self):
        self.assertIsNone(parse_known_error(500, {"error": {"message": "missing"}}))

    def test_truthy_string_cannot_enable_retry(self):
        result = parse_known_error(
            503, {"error": {"code": "SERVICE_UNAVAILABLE", "retryable": "true"}}
        )
        self.assertFalse(result.retryable)

    def test_denial_is_not_retryable_even_if_upstream_claims_it(self):
        result = parse_known_error(
            403, {"error": {"code": "NOT_AUTHORIZED", "retryable": True}}
        )
        self.assertFalse(result.retryable)

    def test_http_400_is_not_assumed_transient(self):
        result = parse_known_error(
            400, {"error": {"code": "SERVICE_UNAVAILABLE", "retryable": True}}
        )
        self.assertFalse(result.retryable)

    def test_boolean_status_not_accepted_as_integer(self):
        self.assertIsNone(parse_known_error(True, {"error": "BAD_INPUT"}))

    def test_success_status_rejects_error_envelope(self):
        self.assertIsNone(parse_known_error(200, {"error": "STATE_CONFLICT"}))

    def test_nested_untrusted_error_code_type(self):
        self.assertIsNone(parse_known_error(500, {"error": {"code": [123]}}))

    def test_serialization_is_stable(self):
        error = parse_known_error(400, {"error": "BAD_INPUT"})
        self.assertEqual(
            error.to_dict(),
            {
                "error": {
                    "code": "BAD_INPUT",
                    "message": "The request could not be accepted.",
                    "retryable": False,
                },
                "status": 400,
            },
        )


if __name__ == "__main__":
    unittest.main()