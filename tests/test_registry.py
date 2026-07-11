import unittest

from lnl_toolbox.registry import Registry


class RegistryTest(unittest.TestCase):
    def test_register_and_build(self) -> None:
        registry = Registry("loss")

        @registry.register("demo")
        def build(value: int) -> int:
            return value * 2

        self.assertEqual(registry.build("demo", value=3), 6)
        self.assertEqual(registry.names(), ("demo",))


if __name__ == "__main__":
    unittest.main()

