.. Multi-Platform Quantum Programming documentation master file, created by
   sphinx-quickstart on Mon Feb 13 11:37:55 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the Multi-Platform Quantum Programming library's documentation!
==========================================================================

This is the reference manual for our Multi-Platform Quantum Programming library 
(in short MPQP). You will find here all the elements needed to be able to use it
fluently.

MPQP was created to solve problems the researchers of our company were facing
when coding quantum algorithms. It is (for now) focused on gate based quantum
computing. The main problem was that, in order to test our algorithms, we had to
redevelop it for each SDK. The decision was then made to make a small library to
drive the other SDKs. Once this decision taken, we realized we could use this
opportunity to make the API more user friendly. During the conception phase we
also realized that many SDKs had bugs slow to fix, so we also decided to bite
the bullet and patch the bugs we could find, until they are fixed on the SDK.
Lastly, after the recent migration from qiskit version `0.x` to `1.x`, we
realized the importance of stability, meaning that we do our best to keep our
library retro-compatible. So, to summarize, this library has 4 goals to improve
on the current available SDKs:

- support for as many SDKs as possible;
- ease of use;
- correctness;
- stability across versions.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   getting-started
   circuit
   instructions
   noise
   execution
   execution-extras
   vqa
   qasm
   tools
   changelog
   examples

Α full index can be found in the :ref:`genindex` page and a map of the module
can be found in the :doc:`all-modules` page.

MPQP is backed by `ColibrITD <https://www.colibritd.com/>`_, and you can chat with
our community on our `Discord <https://discord.gg/yyukutWbzf>`_.