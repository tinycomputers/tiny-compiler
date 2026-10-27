# Security policy

## Supported versions

Only the latest code on `main` is supported. TinyCompiler is an educational compiler and
simulator project; it has no releases yet.

## Reporting a vulnerability

Please do not open a public issue for a security problem.

Report it privately through GitHub: open the repository's **Security** tab and choose
**Report a vulnerability**. The report stays private until a fix is ready.

Include, if you can:

- what you ran (a `.tiny` program, a command, or a page of the Compiler Explorer),
- what you expected and what happened,
- the commit you tested.

You should get an acknowledgement within a few days. A fix will land on `main`, and the report
will be credited there unless you ask otherwise.

## Scope

In scope: the compiler (`tinyc/`), the runners (`runtime/`), the Compiler Explorer (`web/`), and
the build and CI scripts.

Out of scope: the tiny-gpu and tiny-tpu hardware designs, which are separate projects with their
own security contacts, and problems that need a compromised machine to exploit.
