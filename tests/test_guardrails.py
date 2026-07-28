import unittest

from guardrails import rails


class GuardrailsTestCase(unittest.TestCase):
    def test_guard_fails_open_when_guardrails_raise(self):
        class BrokenRails:
            def generate(self, messages):
                raise RuntimeError("guardrails exploded")

        rails._rails = BrokenRails()

        blocked, response = rails.guard("hello")

        self.assertFalse(blocked)
        self.assertIsNone(response)


if __name__ == "__main__":
    unittest.main()
