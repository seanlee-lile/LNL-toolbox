import unittest

from lnl_toolbox.plugins import PluginCatalog
from lnl_toolbox.plugins.builtin import create_builtin_catalog


class PluginCatalogTest(unittest.TestCase):
    def test_capability_discovery(self) -> None:
        catalog = create_builtin_catalog()
        selectors = catalog.find(capability="sample_selection")
        self.assertEqual([(item.kind, item.name) for item in selectors], [("selector", "coteaching_exchange")])

    def test_custom_plugin_does_not_need_lnl_types(self) -> None:
        catalog = PluginCatalog()
        catalog.add("transform", "uppercase", lambda value: value.upper(), capabilities=("text",))
        self.assertEqual(catalog.build("transform", "uppercase", value="hello"), "HELLO")


if __name__ == "__main__":
    unittest.main()

