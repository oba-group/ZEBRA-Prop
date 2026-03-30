"""Tkinter GUI for template-based description generation."""

from __future__ import annotations

from pathlib import Path
import random

import pandas as pd

from .description import (
    extract_feature_labels,
    extract_template_placeholders,
    generate_description_from_dataframe,
    preprocess_dataframe,
    render_template,
)


class TemplateBuilderApp:
    """Interactive GUI for creating templates from descriptor CSV columns."""

    def __init__(
        self,
        tk_module,
        ttk_module,
        filedialog_module,
        messagebox_module,
        data_dir: str | Path,
        input_csv: str | None = None,
    ):
        self.tk = tk_module
        self.ttk = ttk_module
        self.filedialog = filedialog_module
        self.messagebox = messagebox_module

        self.data_dir = Path(data_dir).expanduser().resolve()
        self.description_dir = self.data_dir / "description" / "sample"
        self.descriptor_dir = self.data_dir / "descriptor"

        self.palette = {
            "app_bg": "#f4f7fb",
            "card": "#ffffff",
            "card_alt": "#f8fbff",
            "text": "#13233a",
            "muted": "#59708d",
            "accent": "#ff8f3d",
            "accent_2": "#1ca6ff",
            "accent_3": "#00ba88",
            "line": "#d6e1ef",
            "editor_bg": "#ffffff",
            "editor_fg": "#122235",
            "token_bg": "#ecf6ff",
            "preview_bg": "#f7fbff",
        }

        self.root = self.tk.Tk()
        self.root.title("ZEBRA Template Studio")
        self.root.geometry("1480x920")
        self.root.minsize(1240, 780)
        self.root.configure(bg=self.palette["app_bg"])
        self.root.tk.call("tk", "scaling", 1.15)

        self.search_var = self.tk.StringVar(value="")
        self.status_var = self.tk.StringVar(value="Select one or more descriptor CSV files.")
        self.preview_row_var = self.tk.IntVar(value=0)
        self.formula_simplified_var = self.tk.BooleanVar(value=False)
        self.integerize_var = self.tk.BooleanVar(value=False)
        self.output_name_var = self.tk.StringVar(value="description.csv")
        self.output_path_var = self.tk.StringVar(value="")
        self.loaded_files_var = self.tk.StringVar(value="No files loaded")

        self.drag_token: str | None = None
        self.drag_window = None
        self.last_loaded_dir: Path | None = None

        self.csv_paths: list[Path] = []
        self.csv_frames: dict[Path, pd.DataFrame] = {}
        self.available_tokens: list[str] = []
        self.filtered_tokens: list[str] = []
        self.token_primary_source: dict[str, Path] = {}
        self.token_duplicates: dict[str, list[Path]] = {}

        self._setup_styles()
        self._build_layout()

        self.search_var.trace_add("write", lambda *_: self._refresh_token_list())
        self.output_name_var.trace_add("write", lambda *_: self._update_output_preview())

        if input_csv:
            self._add_csv_paths([Path(input_csv)])

        self._update_output_preview()

    def _setup_styles(self) -> None:
        style = self.ttk.Style()
        style.theme_use("clam")

        style.configure("App.TFrame", background=self.palette["app_bg"])
        style.configure("Card.TFrame", background=self.palette["card"])
        style.configure("CardAlt.TFrame", background=self.palette["card_alt"])

        style.configure(
            "Title.TLabel",
            background=self.palette["card"],
            foreground=self.palette["text"],
            font=("Helvetica", 18, "bold"),
        )
        style.configure(
            "Body.TLabel",
            background=self.palette["card"],
            foreground=self.palette["muted"],
            font=("Helvetica", 12),
        )
        style.configure(
            "BodyAlt.TLabel",
            background=self.palette["card_alt"],
            foreground=self.palette["muted"],
            font=("Helvetica", 12),
        )

        style.configure(
            "Primary.TButton",
            background=self.palette["accent"],
            foreground="#ffffff",
            borderwidth=0,
            padding=(11, 8),
            font=("Helvetica", 12, "bold"),
        )
        style.map("Primary.TButton", background=[("active", "#f0781f")])

        style.configure(
            "Secondary.TButton",
            background=self.palette["accent_2"],
            foreground="#ffffff",
            borderwidth=0,
            padding=(10, 7),
            font=("Helvetica", 11, "bold"),
        )
        style.map("Secondary.TButton", background=[("active", "#0094f0")])

        style.configure(
            "Ghost.TButton",
            background="#eef5ff",
            foreground=self.palette["text"],
            borderwidth=0,
            padding=(10, 7),
            font=("Helvetica", 11),
        )
        style.map("Ghost.TButton", background=[("active", "#e4efff")])

        style.configure(
            "Path.TEntry",
            fieldbackground="#ffffff",
            foreground=self.palette["text"],
            bordercolor=self.palette["line"],
            lightcolor=self.palette["line"],
            darkcolor=self.palette["line"],
            insertcolor=self.palette["accent"],
            padding=6,
            font=("Helvetica", 12),
        )

        style.configure(
            "Switch.TCheckbutton",
            background=self.palette["card_alt"],
            foreground=self.palette["text"],
            font=("Helvetica", 12),
        )

    def _build_layout(self) -> None:
        app = self.ttk.Frame(self.root, style="App.TFrame", padding=10)
        app.pack(fill="both", expand=True)

        self.hero_canvas = self.tk.Canvas(
            app,
            height=120,
            highlightthickness=0,
            bd=0,
            bg=self.palette["card"],
        )
        self.hero_canvas.pack(fill="x", pady=(0, 10))
        self.hero_canvas.bind("<Configure>", self._draw_hero)

        top = self.ttk.Frame(app, style="Card.TFrame", padding=12)
        top.pack(fill="x", pady=(0, 10))

        self.ttk.Button(
            top,
            text="Add CSV Files",
            style="Primary.TButton",
            command=self._load_csv_dialog,
        ).grid(row=0, column=0, padx=(0, 8), pady=(0, 6))

        self.ttk.Button(
            top,
            text="Reset",
            style="Ghost.TButton",
            command=self._reset_csvs,
        ).grid(row=0, column=1, padx=(0, 10), pady=(0, 6))

        self.ttk.Label(top, textvariable=self.loaded_files_var, style="Body.TLabel").grid(
            row=0, column=2, sticky="w", pady=(0, 6)
        )

        self.ttk.Label(
            top,
            text=f"Data Dir: {self._short_path(self.data_dir)}",
            style="Body.TLabel",
        ).grid(row=1, column=0, columnspan=3, sticky="w")

        top.columnconfigure(2, weight=1)

        body = self.ttk.PanedWindow(app, orient="horizontal")
        body.pack(fill="both", expand=True)

        self.left_panel = self.ttk.Frame(body, style="Card.TFrame", padding=12)
        self.center_panel = self.ttk.Frame(body, style="CardAlt.TFrame", padding=12)
        self.right_panel = self.ttk.Frame(body, style="Card.TFrame", padding=12)

        body.add(self.left_panel, weight=1)
        body.add(self.center_panel, weight=2)
        body.add(self.right_panel, weight=1)

        self._build_left_panel()
        self._build_center_panel()
        self._build_right_panel()

        status = self.ttk.Frame(app, style="Card.TFrame", padding=(10, 7))
        status.pack(fill="x", pady=(10, 0))
        self.ttk.Label(status, textvariable=self.status_var, style="Body.TLabel").pack(
            side="left"
        )

    def _draw_hero(self, event) -> None:
        canvas = self.hero_canvas
        canvas.delete("all")

        width = max(event.width, 1)
        height = max(event.height, 1)

        canvas.create_rectangle(0, 0, width, height, fill=self.palette["card"], outline="")

        stripe_span = 72
        stripe_width = 34
        x = -height
        while x < width + height:
            canvas.create_polygon(
                x,
                0,
                x + stripe_width,
                0,
                x + stripe_width + height,
                height,
                x + height,
                height,
                fill="#f4f4f4",
                outline="",
                stipple="gray25",
            )
            x += stripe_span

        canvas.create_oval(width - 260, -30, width - 110, 120, fill="#ffe2cf", outline="")
        canvas.create_oval(width - 170, 20, width - 20, 170, fill="#d9f1ff", outline="")

        canvas.create_text(
            22,
            40,
            anchor="w",
            text="ZEBRA Template Studio",
            fill=self.palette["text"],
            font=("Helvetica", 29, "bold"),
        )
        canvas.create_text(
            22,
            76,
            anchor="w",
            text=(
                "Build polished description templates with drag-and-drop tokens, "
                "multi-CSV blending, and smart draft generation."
            ),
            fill=self.palette["muted"],
            font=("Helvetica", 14),
        )

    def _build_left_panel(self) -> None:
        self.ttk.Label(self.left_panel, text="Token Library", style="Title.TLabel").pack(
            anchor="w"
        )
        self.ttk.Label(
            self.left_panel,
            text="Drag tokens into the editor or double-click to insert.",
            style="Body.TLabel",
        ).pack(anchor="w", pady=(2, 10))

        self.ttk.Entry(
            self.left_panel,
            textvariable=self.search_var,
            style="Path.TEntry",
        ).pack(fill="x", pady=(0, 8))

        formula_bar = self.ttk.Frame(self.left_panel, style="Card.TFrame")
        formula_bar.pack(fill="x", pady=(0, 8))

        formula_chip = self.ttk.Button(
            formula_bar,
            text="{{formula}}",
            style="Ghost.TButton",
            command=lambda: self._insert_placeholder("formula"),
        )
        formula_chip.pack(side="left")
        formula_chip.bind(
            "<ButtonPress-1>",
            lambda e, token="formula": self._start_drag_token(e, token),
        )
        formula_chip.bind("<B1-Motion>", self._drag_token_motion)
        formula_chip.bind("<ButtonRelease-1>", self._drop_token)

        list_wrap = self.ttk.Frame(self.left_panel, style="Card.TFrame")
        list_wrap.pack(fill="both", expand=True)

        scroll = self.ttk.Scrollbar(list_wrap)
        scroll.pack(side="right", fill="y")

        self.token_list = self.tk.Listbox(
            list_wrap,
            bg=self.palette["token_bg"],
            fg=self.palette["text"],
            selectbackground="#d6ecff",
            selectforeground=self.palette["text"],
            highlightthickness=1,
            highlightbackground=self.palette["line"],
            highlightcolor=self.palette["accent_2"],
            activestyle="none",
            font=("Helvetica", 12),
            yscrollcommand=scroll.set,
        )
        self.token_list.pack(side="left", fill="both", expand=True)
        scroll.config(command=self.token_list.yview)

        self.token_list.bind("<Double-Button-1>", self._insert_selected_token)
        self.token_list.bind("<ButtonPress-1>", self._start_drag_from_list)
        self.token_list.bind("<B1-Motion>", self._drag_token_motion)
        self.token_list.bind("<ButtonRelease-1>", self._drop_token)

    def _build_center_panel(self) -> None:
        self.ttk.Label(
            self.center_panel,
            text="Template Composer",
            style="Title.TLabel",
        ).pack(anchor="w")
        self.ttk.Label(
            self.center_panel,
            text="Focus on writing. Use formula token + descriptor tokens.",
            style="BodyAlt.TLabel",
        ).pack(anchor="w", pady=(2, 10))

        bar = self.ttk.Frame(self.center_panel, style="CardAlt.TFrame")
        bar.pack(fill="x", pady=(0, 8))

        self.ttk.Button(
            bar,
            text="Insert Formula",
            style="Ghost.TButton",
            command=lambda: self._insert_placeholder("formula"),
        ).pack(side="left", padx=(0, 6))

        self.ttk.Button(
            bar,
            text="Random Draft",
            style="Secondary.TButton",
            command=self._generate_random_template,
        ).pack(side="left", padx=(0, 6))

        self.ttk.Button(
            bar,
            text="Clear",
            style="Ghost.TButton",
            command=self._clear_template,
        ).pack(side="left", padx=(0, 6))

        self.ttk.Button(
            bar,
            text="Undo",
            style="Ghost.TButton",
            command=self._undo_template,
        ).pack(side="left")

        editor_wrap = self.ttk.Frame(self.center_panel, style="CardAlt.TFrame")
        editor_wrap.pack(fill="both", expand=True)

        yscroll = self.ttk.Scrollbar(editor_wrap)
        yscroll.pack(side="right", fill="y")

        self.template_text = self.tk.Text(
            editor_wrap,
            wrap="word",
            undo=True,
            bg=self.palette["editor_bg"],
            fg=self.palette["editor_fg"],
            insertbackground=self.palette["accent"],
            selectbackground="#d6ecff",
            selectforeground=self.palette["text"],
            highlightthickness=1,
            highlightbackground=self.palette["line"],
            highlightcolor=self.palette["accent_2"],
            font=("Menlo", 15),
            yscrollcommand=yscroll.set,
            padx=14,
            pady=14,
        )
        self.template_text.pack(fill="both", expand=True)
        yscroll.config(command=self.template_text.yview)

        self.template_text.bind("<KeyRelease>", lambda _: self._refresh_preview())

    def _build_right_panel(self) -> None:
        self.ttk.Label(self.right_panel, text="Preview & Export", style="Title.TLabel").pack(
            anchor="w"
        )
        self.ttk.Label(
            self.right_panel,
            text="Multi-CSV preview with optional simplified formula and integerize.",
            style="Body.TLabel",
        ).pack(anchor="w", pady=(2, 8))

        settings = self.ttk.Frame(self.right_panel, style="Card.TFrame")
        settings.pack(fill="x", pady=(0, 8))

        self.ttk.Checkbutton(
            settings,
            text="Simplified formula for all rows",
            variable=self.formula_simplified_var,
            style="Switch.TCheckbutton",
            command=self._refresh_preview,
        ).pack(anchor="w", pady=(0, 3))

        self.ttk.Checkbutton(
            settings,
            text="Integerize all numeric labels",
            variable=self.integerize_var,
            style="Switch.TCheckbutton",
            command=self._refresh_preview,
        ).pack(anchor="w")

        row_ctrl = self.ttk.Frame(self.right_panel, style="Card.TFrame")
        row_ctrl.pack(fill="x", pady=(0, 8))
        self.ttk.Label(row_ctrl, text="Preview Row", style="Body.TLabel").pack(side="left")

        self.row_spinbox = self.tk.Spinbox(
            row_ctrl,
            from_=0,
            to=0,
            textvariable=self.preview_row_var,
            width=8,
            command=self._refresh_preview,
            bg="#ffffff",
            fg=self.palette["text"],
            insertbackground=self.palette["accent"],
            relief="flat",
            highlightthickness=1,
            highlightbackground=self.palette["line"],
            highlightcolor=self.palette["accent_2"],
            buttonbackground="#eef5ff",
        )
        self.row_spinbox.pack(side="left", padx=(8, 0))
        self.row_spinbox.bind("<Return>", lambda _: self._refresh_preview())
        self.row_spinbox.bind("<FocusOut>", lambda _: self._refresh_preview())

        preview_wrap = self.ttk.Frame(self.right_panel, style="Card.TFrame")
        preview_wrap.pack(fill="both", expand=True)

        self.preview_text = self.tk.Text(
            preview_wrap,
            wrap="word",
            height=12,
            state="disabled",
            bg=self.palette["preview_bg"],
            fg=self.palette["text"],
            highlightthickness=1,
            highlightbackground=self.palette["line"],
            font=("Helvetica", 16),
            padx=10,
            pady=10,
        )
        self.preview_text.pack(fill="both", expand=True)

        self.diagnostics_text = self.tk.Text(
            self.right_panel,
            wrap="word",
            height=7,
            state="disabled",
            bg="#f0f7ff",
            fg=self.palette["muted"],
            highlightthickness=1,
            highlightbackground=self.palette["line"],
            font=("Helvetica", 12),
            padx=10,
            pady=8,
        )
        self.diagnostics_text.pack(fill="x", pady=(8, 8))

        output = self.ttk.Frame(self.right_panel, style="Card.TFrame")
        output.pack(fill="x")
        self.ttk.Label(output, text="Output File Name", style="Body.TLabel").pack(anchor="w")
        self.ttk.Entry(
            output,
            textvariable=self.output_name_var,
            style="Path.TEntry",
        ).pack(fill="x", pady=(4, 4))
        self.ttk.Label(output, textvariable=self.output_path_var, style="Body.TLabel").pack(
            anchor="w"
        )

        self.ttk.Button(
            self.right_panel,
            text="Export CSV",
            style="Primary.TButton",
            command=self._export_csv,
        ).pack(fill="x", pady=(10, 0))

    def _short_path(self, path: Path) -> str:
        target = path.expanduser().resolve()

        bases = []
        cwd = Path.cwd().resolve()
        bases.append(cwd)
        if cwd.parent != cwd:
            bases.append(cwd.parent)
        if cwd.parent.parent != cwd.parent:
            bases.append(cwd.parent.parent)

        for base in reversed(bases):
            try:
                rel = target.relative_to(base)
                return str(rel)
            except Exception:
                continue
        return str(target)

    def _normalize_output_name(self, *, commit: bool = False) -> str:
        name = self.output_name_var.get().strip() or "description.csv"
        if not name.lower().endswith(".csv"):
            name += ".csv"
        name = Path(name).name
        if commit:
            self.output_name_var.set(name)
        return name

    def _current_description_dir(self) -> Path:
        if self.last_loaded_dir is not None:
            # Create `description/sample/` as a sibling of the last loaded CSV directory.
            return self.last_loaded_dir.parent / "description" / "sample"
        return self.description_dir

    def _resolve_output_path(self, *, commit_output_name: bool = False) -> Path:
        filename = self._normalize_output_name(commit=commit_output_name)
        return self._current_description_dir() / filename

    def _update_output_preview(self) -> None:
        out_path = self._resolve_output_path()
        self.output_path_var.set(f"Save to: {self._short_path(out_path)}")

    def _load_csv_dialog(self) -> None:
        files = self.filedialog.askopenfilenames(
            title="Select descriptor CSV files",
            initialdir=str(self.descriptor_dir if self.descriptor_dir.exists() else self.data_dir),
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not files:
            return
        self._add_csv_paths([Path(p) for p in files])

    def _read_csv(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        if "material_id" in df.columns and "id" not in df.columns:
            df = df.rename(columns={"material_id": "id"})

        missing = {"id", "formula"} - set(df.columns)
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"{path.name} is missing required columns: {missing_text}")

        return df

    def _add_csv_paths(self, paths: list[Path]) -> None:
        added: list[Path] = []
        selected_last: Path | None = None
        for path in paths:
            resolved = path.expanduser().resolve()
            selected_last = resolved
            if resolved in self.csv_paths:
                continue
            try:
                df = self._read_csv(resolved)
            except Exception as exc:
                self.messagebox.showerror("CSV Error", str(exc))
                continue
            self.csv_paths.append(resolved)
            self.csv_frames[resolved] = df
            added.append(resolved)

        if selected_last is not None:
            self.last_loaded_dir = selected_last.parent
            self._update_output_preview()

        if not added and self.csv_paths:
            self.status_var.set("No new files were added.")
            return

        self._rebuild_token_sources()
        self._refresh_token_list()
        self._update_loaded_files_label()

        max_row = max(len(self._build_merged_dataframe(self.available_tokens)) - 1, 0)
        self.row_spinbox.config(from_=0, to=max_row)
        self.preview_row_var.set(min(self.preview_row_var.get(), max_row))

        if added:
            self.status_var.set(f"Added {len(added)} file(s).")
        self._refresh_preview()

    def _reset_csvs(self) -> None:
        self.csv_paths.clear()
        self.csv_frames.clear()
        self.available_tokens.clear()
        self.filtered_tokens.clear()
        self.token_primary_source.clear()
        self.token_duplicates.clear()
        self.last_loaded_dir = None
        self.token_list.delete(0, self.tk.END)
        self.loaded_files_var.set("No files loaded")
        self.preview_row_var.set(0)
        self.row_spinbox.config(from_=0, to=0)
        self.status_var.set("CSV selection cleared.")
        self._update_output_preview()
        self._refresh_preview()

    def _update_loaded_files_label(self) -> None:
        names = [p.name for p in self.csv_paths]
        if not names:
            self.loaded_files_var.set("No files loaded")
            return
        label_count = len([t for t in self.available_tokens if t != "formula"])
        self.loaded_files_var.set(
            f"{len(names)} files | {label_count} labels | " + ", ".join(names[:3]) + (" ..." if len(names) > 3 else "")
        )

    def _rebuild_token_sources(self) -> None:
        self.available_tokens = ["formula"]
        self.token_primary_source = {}
        self.token_duplicates = {}

        for path in self.csv_paths:
            df = self.csv_frames[path]
            labels = extract_feature_labels(df)
            for label in labels:
                if label not in self.token_primary_source:
                    self.token_primary_source[label] = path
                    self.available_tokens.append(label)
                else:
                    self.token_duplicates.setdefault(label, [self.token_primary_source[label]])
                    self.token_duplicates[label].append(path)

    def _refresh_token_list(self) -> None:
        query = self.search_var.get().strip().lower()
        tokens = [t for t in self.available_tokens if t != "formula"]

        if query:
            self.filtered_tokens = [t for t in tokens if query in t.lower()]
        else:
            self.filtered_tokens = tokens

        self.token_list.delete(0, self.tk.END)
        for token in self.filtered_tokens:
            self.token_list.insert(self.tk.END, token)

    def _insert_selected_token(self, _event=None) -> None:
        selection = self.token_list.curselection()
        if not selection:
            return
        token = self.token_list.get(selection[0])
        self._insert_placeholder(token)

    def _insert_placeholder(self, token: str, index: str = "insert") -> None:
        placeholder = f"{{{{{token}}}}}"
        try:
            prev_char = self.template_text.get(f"{index} -1c")
        except Exception:
            prev_char = ""

        needs_space = bool(prev_char and not prev_char.isspace() and prev_char not in "([{")
        if needs_space:
            self.template_text.insert(index, " ")
            index = self.template_text.index(f"{index} +1c")

        self.template_text.insert(index, placeholder)
        self.template_text.focus_set()
        self._refresh_preview()

    def _clear_template(self) -> None:
        self.template_text.delete("1.0", self.tk.END)
        self._refresh_preview()

    def _undo_template(self) -> None:
        try:
            self.template_text.edit_undo()
        except Exception:
            return
        self._refresh_preview()

    def _generate_random_template(self) -> None:
        labels = [t for t in self.available_tokens if t != "formula"]
        if not labels:
            self.status_var.set("Load descriptor CSVs before generating a random draft.")
            return

        sample_size = min(len(labels), random.randint(4, 10))
        sample_size = max(2, sample_size)
        chosen = random.sample(labels, k=sample_size)

        def _slot(label: str) -> str:
            return f"{{{{{label}}}}}"

        def _pretty(label: str) -> str:
            return label.replace("_", " ")

        first = chosen[0]
        second = chosen[1] if len(chosen) > 1 else chosen[0]

        opening_pool = [
            "In this study, {{{{formula}}}} is characterized by {a} = {A} and {b} = {B}.",
            "The present analysis indicates that {{{{formula}}}} exhibits {a} = {A} together with {b} = {B}.",
            "From a descriptor perspective, {{{{formula}}}} is represented by {a} = {A} and {b} = {B}.",
            "The material {{{{formula}}}} can be quantitatively described using {a} = {A} and {b} = {B}.",
            "Within this framework, {{{{formula}}}} shows {a} = {A} alongside {b} = {B}.",
        ]
        opening = random.choice(opening_pool).format(
            a=_pretty(first),
            A=_slot(first),
            b=_pretty(second),
            B=_slot(second),
        )

        sentences: list[str] = [opening]

        cursor = 2
        while cursor < len(chosen):
            picked = chosen[cursor : cursor + 2]
            cursor += 2
            if len(picked) == 1:
                label = picked[0]
                single_pool = [
                    "Additionally, {x} is quantified as {X}.",
                    "Moreover, {x} is observed at {X}.",
                    "In parallel, {x} takes the value {X}.",
                ]
                sentences.append(
                    random.choice(single_pool).format(
                        x=_pretty(label),
                        X=_slot(label),
                    )
                )
            else:
                a, b = picked
                pair_pool = [
                    "Furthermore, {a} and {b} are observed as {A} and {B}, respectively.",
                    "Concurrently, {a} = {A} and {b} = {B} contribute to the descriptor profile.",
                    "The measured descriptor set also includes {a} = {A} and {b} = {B}.",
                ]
                sentences.append(
                    random.choice(pair_pool).format(
                        a=_pretty(a),
                        A=_slot(a),
                        b=_pretty(b),
                        B=_slot(b),
                    )
                )

        closing_pool = [
            "Collectively, these descriptor values provide a compact representation of the compositional and physicochemical characteristics of {{formula}}.",
            "Overall, this descriptor combination summarizes the key material signatures of {{formula}}.",
            "Taken together, the descriptor pattern offers an integrated view of {{formula}}.",
        ]
        sentences.append(random.choice(closing_pool))

        template = " ".join(sentences)

        self.template_text.delete("1.0", self.tk.END)
        self.template_text.insert("1.0", template)
        self.status_var.set("Generated a random paper-style template draft.")
        self._refresh_preview()

    def _start_drag_from_list(self, event) -> None:
        index = self.token_list.nearest(event.y)
        if index < 0:
            self.drag_token = None
            return

        self.token_list.selection_clear(0, self.tk.END)
        self.token_list.selection_set(index)
        token = self.token_list.get(index)
        self._start_drag_token(event, token)

    def _start_drag_token(self, event, token: str) -> None:
        self.drag_token = token

        if self.drag_window is not None:
            self.drag_window.destroy()

        self.drag_window = self.tk.Toplevel(self.root)
        self.drag_window.overrideredirect(True)
        self.drag_window.configure(bg=self.palette["accent_2"])
        self.drag_window.attributes("-alpha", 0.9)

        label = self.tk.Label(
            self.drag_window,
            text=f"{{{{{token}}}}}",
            bg=self.palette["accent_2"],
            fg="#ffffff",
            padx=8,
            pady=4,
            font=("Helvetica", 12, "bold"),
        )
        label.pack()
        self._move_drag_window(event)

    def _move_drag_window(self, event) -> None:
        if self.drag_window is None:
            return
        self.drag_window.geometry(f"+{event.x_root + 14}+{event.y_root + 14}")

    def _drag_token_motion(self, event) -> None:
        if not self.drag_token:
            return

        self._move_drag_window(event)

        target = self.root.winfo_containing(event.x_root, event.y_root)
        if self._is_descendant(target, self.template_text):
            x = event.x_root - self.template_text.winfo_rootx()
            y = event.y_root - self.template_text.winfo_rooty()
            index = self.template_text.index(f"@{x},{y}")
            self.template_text.mark_set("insert", index)

    def _drop_token(self, event) -> None:
        if not self.drag_token:
            return

        target = self.root.winfo_containing(event.x_root, event.y_root)
        if self._is_descendant(target, self.template_text):
            x = event.x_root - self.template_text.winfo_rootx()
            y = event.y_root - self.template_text.winfo_rooty()
            index = self.template_text.index(f"@{x},{y}")
            self._insert_placeholder(self.drag_token, index=index)

        self.drag_token = None
        if self.drag_window is not None:
            self.drag_window.destroy()
            self.drag_window = None

    def _is_descendant(self, widget, ancestor) -> bool:
        while widget is not None:
            if widget == ancestor:
                return True
            widget = getattr(widget, "master", None)
        return False

    def _build_merged_dataframe(
        self,
        tokens: list[str],
        frames_override: dict[Path, pd.DataFrame] | None = None,
    ) -> pd.DataFrame:
        if not self.csv_paths:
            return pd.DataFrame(columns=["id", "formula"])

        frames = frames_override if frames_override is not None else self.csv_frames

        base = None
        for path in self.csv_paths:
            if path not in frames:
                continue
            df = frames[path]
            key_df = df[["id", "formula"]].copy()
            if base is None:
                base = key_df
            else:
                base = base.merge(key_df, on=["id", "formula"], how="outer")

        if base is None:
            return pd.DataFrame(columns=["id", "formula"])

        for token in tokens:
            if token == "formula":
                continue

            source = self.token_primary_source.get(token)
            if source is None or source not in frames:
                continue

            src_df = frames[source]
            if token not in src_df.columns:
                continue

            column_df = src_df[["id", "formula", token]].copy()
            base = base.merge(column_df, on=["id", "formula"], how="left")

        return base

    def _refresh_preview(self) -> None:
        if not self.csv_paths:
            self._set_text(self.preview_text, "No CSV loaded.")
            self._set_text(
                self.diagnostics_text,
                "Diagnostics:\n- Placeholders: none\n- CSV files: none\n- Missing columns: n/a",
            )
            return

        preview_df = self._build_merged_dataframe(self.available_tokens)
        preview_df = preprocess_dataframe(
            preview_df,
            formula_simplified=self.formula_simplified_var.get(),
            integerize=self.integerize_var.get(),
        )

        if preview_df.empty:
            self._set_text(self.preview_text, "No rows after merge.")
            return

        max_row = len(preview_df) - 1
        self.row_spinbox.config(from_=0, to=max_row)

        try:
            row_idx = int(self.preview_row_var.get())
        except Exception:
            row_idx = 0
        row_idx = min(max(row_idx, 0), max_row)
        self.preview_row_var.set(row_idx)

        template = self.template_text.get("1.0", self.tk.END).strip()
        row = preview_df.iloc[row_idx]

        if template:
            preview = render_template(row, template)
        else:
            default_df = generate_description_from_dataframe(
                preview_df.iloc[[row_idx]],
                formula_simplified=False,
                integerize=False,
            )
            preview = str(default_df.iloc[0]["description"])

        self._set_text(self.preview_text, preview)

        placeholders = extract_template_placeholders(template) if template else []
        missing = []
        if placeholders:
            existing = set(preview_df.columns)
            missing = [token for token in placeholders if token != "formula" and token not in existing]

        duplicate_tokens = sorted(set(placeholders) & set(self.token_duplicates))

        diag = ["Diagnostics:"]
        diag.append(
            "- Placeholders: " + (", ".join(placeholders) if placeholders else "none (default mode)")
        )
        diag.append(
            "- CSV files: " + ", ".join(path.name for path in self.csv_paths)
        )
        diag.append(
            "- Missing columns: " + (", ".join(missing) if missing else "none")
        )
        if duplicate_tokens:
            diag.append(
                "- Duplicate labels detected: " + ", ".join(duplicate_tokens)
            )
        diag.append(f"- Preview row: {row_idx} / {max_row}")

        self._set_text(self.diagnostics_text, "\n".join(diag))

    def _set_text(self, widget, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", self.tk.END)
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    def _reload_frames(self, paths: list[Path]) -> dict[Path, pd.DataFrame]:
        reloaded: dict[Path, pd.DataFrame] = {}
        for path in paths:
            reloaded[path] = self._read_csv(path)
        return reloaded

    def _determine_export_tokens_and_paths(self, template: str) -> tuple[list[str], list[Path]]:
        if template:
            placeholders = extract_template_placeholders(template)
            tokens = [token for token in placeholders if token != "formula"]
            if not tokens:
                return [], self.csv_paths
            used_paths: list[Path] = []
            seen = set()
            for token in tokens:
                source = self.token_primary_source.get(token)
                if source is not None and source not in seen:
                    seen.add(source)
                    used_paths.append(source)
            return tokens, used_paths

        tokens = [token for token in self.available_tokens if token != "formula"]
        return tokens, list(self.csv_paths)

    def _export_csv(self) -> None:
        if not self.csv_paths:
            self.messagebox.showwarning("No Data", "Please load descriptor CSV files first.")
            return

        template = self.template_text.get("1.0", self.tk.END).strip()
        tokens, used_paths = self._determine_export_tokens_and_paths(template)

        if not used_paths:
            self.messagebox.showwarning(
                "No Matching Sources",
                "No CSV sources matched the current template placeholders.",
            )
            return

        try:
            reloaded = self._reload_frames(used_paths)
        except Exception as exc:
            self.messagebox.showerror("Read Error", f"Failed to reload CSV files:\n{exc}")
            return

        merged = self._build_merged_dataframe(tokens or self.available_tokens, frames_override=reloaded)

        out_path = self._resolve_output_path(commit_output_name=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        output_df = generate_description_from_dataframe(
            merged,
            template=template if template else None,
            formula_simplified=self.formula_simplified_var.get(),
            integerize=self.integerize_var.get(),
        )
        output_df.to_csv(out_path, index=False)

        self.status_var.set(
            f"Exported {len(output_df)} rows using {len(used_paths)} source file(s)."
        )
        self._update_output_preview()
        self.messagebox.showinfo("Export Complete", f"Saved:\n{out_path}")

    def run(self) -> None:
        self.root.mainloop()


def run_gui(
    data_dir: str | Path,
    input_csv: str | None = None,
) -> None:
    """Launch template-builder GUI."""
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except Exception as exc:
        raise RuntimeError(
            "GUI mode requires tkinter. Use normal mode if tkinter is unavailable."
        ) from exc

    app = TemplateBuilderApp(
        tk_module=tk,
        ttk_module=ttk,
        filedialog_module=filedialog,
        messagebox_module=messagebox,
        data_dir=data_dir,
        input_csv=input_csv,
    )
    app.run()
