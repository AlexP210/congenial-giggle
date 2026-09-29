
class BaseClass:
	"""Base class that all classes inherit from to ensure that we never call object.__init__(cfg)"""
	def __init__(self, cfg):
		super().__init__()