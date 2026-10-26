// The Python compiler, running in Pyodide inside a worker so the page never blocks.
// Pinned Pyodide release; the first analyze() downloads it (numpy is loaded as a package).
importScripts("https://cdn.jsdelivr.net/pyodide/v0.27.7/full/pyodide.js");

const ready = (async () => {
  const py = await loadPyodide({ indexURL: "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/" });
  await py.loadPackage("numpy");
  const { files } = await (await fetch("/py/bundle.json")).json();
  const root = "/home/pyodide/site";
  for (const [path, text] of Object.entries(files)) {
    const full = `${root}/${path}`;
    py.FS.mkdirTree(full.slice(0, full.lastIndexOf("/")));
    py.FS.writeFile(full, text);
  }
  py.runPython(`import sys; sys.path.insert(0, "${root}")`);
  py.runPython("import runtime.web_api as api");
  return py;
})();

onmessage = async (event) => {
  const { id, source, opt, seed } = event.data;
  try {
    const py = await ready;
    py.globals.set("SRC", source);
    py.globals.set("OPT", opt);
    py.globals.set("SEED", seed);
    const text = py.runPython("api.analyze(SRC, OPT, SEED)");
    postMessage({ id, ok: true, result: JSON.parse(text) });
  } catch (err) {
    postMessage({ id, ok: false, error: String((err && err.message) || err) });
  }
};
