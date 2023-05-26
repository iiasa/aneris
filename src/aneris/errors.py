class AmbiguousHarmonisationMethod(ValueError):
    """
    Error raised when harmonisation methods are ambiguous.
    """


class MissingHistoricalError(ValueError):
    """
    Error raised when historical data is missing.
    """

class MissingProxyError(ValueError):
    """
    Error raised when required proxy data is missing.
    """


class MissingScenarioError(ValueError):
    """
    Error raised when scenario data is missing.
    """


class MissingHarmonisationYear(ValueError):
    """
    Error raised when the harmonisation year is missing.
    """
