"""
Percepta Python Backend Service
Real-time contextual Q&A for Twitch streams powered by RAG and live transcription
"""

__version__ = "0.1.0"

# Compatibility shim: some tools (e.g., pytest internals/plugins) may access
# py.path.local from the legacy 'py' library. Since this project package is
# also named 'py', provide a minimal stub to prevent attribute errors.
class _CompatPath:
	def local(self, p=None):
		return p

# Expose as py.path.local
path = _CompatPath()
