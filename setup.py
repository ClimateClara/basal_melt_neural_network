from setuptools import setup


setup(
    use_scm_version={
        "write_to": "basal_melt_neural_networks/_version.py",
        "write_to_template": '__version__ = "{version}"',
        "tag_regex": r"^(?P<prefix>v)?(?P<version>[^\+]+)(?P<suffix>.*)?$",
    }
)
