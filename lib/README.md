# Vendored build dependency

## `utilities-5.0.39P2.jar`

`com.compomics:utilities:5.0.39P2` is a **patched** build of compomics-utilities. The
public compomics/genesis Maven repositories host `5.0.39` but not the `P2` patch, so a
clean checkout (CI or a new contributor) cannot resolve it from any repository.

It is vendored here, with a matching exception in `.gitignore` (which otherwise ignores
`*.jar`), so the project can be built without manually running `install:install-file`.
The `build-installer.yml` workflow installs it into the runner's local Maven repository
before `mvn package`.

**This is a stop-gap.** The clean long-term fix is to publish this jar to a
lab-controlled Maven repository and resolve it as a normal dependency, then remove
`lib/` and the install step from the workflow.
