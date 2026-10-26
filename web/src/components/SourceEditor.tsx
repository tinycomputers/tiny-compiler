import { useEffect, useRef } from "react";
import { EditorView, basicSetup } from "codemirror";
import { EditorState } from "@codemirror/state";

interface Props {
  value: string;
  onChange: (value: string) => void;
  label: string;
}

// A plain CodeMirror editor, controlled by `value`: edits call `onChange`, and changes to
// `value` from outside (choosing an example) replace the document.
export default function SourceEditor({ value, onChange, label }: Props) {
  const host = useRef<HTMLDivElement>(null);
  const view = useRef<EditorView | null>(null);
  const onChangeRef = useRef(onChange);
  onChangeRef.current = onChange;

  useEffect(() => {
    if (!host.current) return;
    view.current = new EditorView({
      parent: host.current,
      state: EditorState.create({
        doc: value,
        extensions: [
          basicSetup,
          EditorView.updateListener.of((update) => {
            if (update.docChanged) onChangeRef.current(update.state.doc.toString());
          }),
          EditorView.contentAttributes.of({ "aria-label": label, spellcheck: "false" }),
          EditorView.theme({
            "&": { backgroundColor: "var(--panel)", color: "var(--fg)", fontSize: "13px" },
            ".cm-content": { fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace" },
            ".cm-gutters": { backgroundColor: "var(--panel)", color: "var(--muted)", border: "none" },
            "&.cm-focused": { outline: "1px solid var(--accent)" },
          }),
        ],
      }),
    });
    return () => view.current?.destroy();
    // The editor is created once; later value changes go through the effect below.
  }, []);

  useEffect(() => {
    const current = view.current;
    if (current && current.state.doc.toString() !== value) {
      current.dispatch({ changes: { from: 0, to: current.state.doc.length, insert: value } });
    }
  }, [value]);

  return <div ref={host} style={{ minHeight: 360, border: "1px solid var(--line)", borderRadius: 6, overflow: "hidden" }} />;
}
