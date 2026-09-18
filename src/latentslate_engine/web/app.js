import { appendCollectionItem, moveCollectionItem, removeCollectionItem } from "./collections.js";

const $ = (id) => document.getElementById(id);
const state = {
  builtins: [], users: [], operations: new Map(), document: null,
  builtinKey: null, revision: null, hash: null, dirty: false, busy: false,
  validation: null, token: sessionStorage.getItem("latentslate.authoring.token") || "",
  picker: null, searchSequence: 0, searchTimer: null,
  imports: [], importBusy: false,
  publication: null,
  libraryCollapsed: { users: false }, familyCollapsed: new Map(),
  hfPicker: null, pinnedSource: null, artifactTask: null, materializationDocument: null, materializationPlan: null,
  civitaiPicker: null, civitaiVersion: null, civitaiPinned: null,
  downloadRecipes: [], downloadPlan: null, downloadSequence: 0,
};

// Browser-native source-aware JSON keeps the existing unsigned 64-bit integer
// domain intact. Number coercion must not silently round saved seeds/bounds.
function parseJSON(text) {
  return JSON.parse(text, (key, value, context) => {
    if (typeof value === "number" && Number.isInteger(value) && !Number.isSafeInteger(value)) {
      if (!context?.source) throw new Error("This browser cannot preserve large integers. Please use a current browser.");
      if (/^-?\d+$/.test(context.source)) return BigInt(context.source);
    }
    return value;
  });
}

function stringifyJSON(value) {
  return JSON.stringify(value, (key, item) => typeof item === "bigint" ? JSON.rawJSON(item.toString()) : item);
}

function element(tag, attributes = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attributes)) {
    if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (["value", "checked", "disabled", "hidden", "readOnly"].includes(key)) node[key] = value;
    else node.setAttribute(key, value);
  }
  for (const child of children) node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  return node;
}

function label(key) {
  return key.replaceAll("_", " ").replace(/\bloras\b/gi, "LoRAs").replace(/\bcfg\b/gi, "CFG").replace(/^./, (letter) => letter.toUpperCase());
}

function fieldLabel(descriptor) {
  return descriptor.presentation?.label || label(descriptor.key);
}

function pill(text, kind = "") {
  return element("span", { class: `status-pill ${kind}`, text });
}

function notice(message = "", error = false, action = null) {
  const node = $("notice");
  node.hidden = !message;
  node.className = `notice topbar-notice${error ? " error" : ""}`;
  node.replaceChildren(element("span", { class: "notice-message", text: message }));
  if (action) node.append(element("button", { class: "secondary", text: action.text, onclick: action.run }));
}

async function api(path, options = {}, textResponse = false) {
  const headers = {};
  if (state.token) headers.Authorization = `Bearer ${state.token}`;
  if (options.body !== undefined) headers["Content-Type"] = "application/json";
  let response;
  try {
    response = await fetch(`/v1/authoring${path}`, { ...options, headers, body: options.body === undefined ? undefined : typeof options.body === "string" ? options.body : stringifyJSON(options.body) });
  } catch {
    throw new Error("Cannot reach this Engine. Check the connection and try again.");
  }
  const text = await response.text();
  if (response.ok && textResponse) return text;
  let data;
  try { data = parseJSON(text); } catch (error) {
    if (response.ok) throw error;
    data = {};
  }
  if (!response.ok) {
    const error = new Error(data.error?.message || `Engine request failed (${response.status})`);
    error.status = response.status;
    error.data = data;
    throw error;
  }
  return data;
}

function showError(error) {
  if (error.data?.validation) {
    state.validation = error.data.validation;
    renderValidation(true);
  }
  if (error.status === 401) {
    $("connection-status").textContent = "Authentication required";
    $("connection-button").classList.remove("connected");
    $("connection-error").textContent = "This Engine requires a bearer token.";
    openConnection();
  } else if (error.status === 409 && state.document && !state.builtinKey) {
    notice(`${error.message}. Your unsaved edits are still here.`, true, { text: "Reload latest", run: reloadLatest });
  } else notice(error.message, true);
}

async function work(action) {
  if (state.busy) return;
  state.busy = true;
  document.querySelector(".workspace").inert = true;
  updateToolbar();
  try { await action(); } catch (error) { showError(error); }
  finally {
    state.busy = false;
    document.querySelector(".workspace").inert = false;
    updateToolbar();
  }
}

function updateToolbar() {
  const builtin = Boolean(state.builtinKey);
  $("duplicate-button").hidden = !builtin;
  $("save-button").hidden = builtin;
  $("export-button").hidden = builtin || !state.document;
  $("delete-button").hidden = builtin || !state.document;
  $("delete-button").disabled = state.busy;
  $("export-button").disabled = state.busy || state.dirty;
  $("save-button").disabled = state.busy || !state.dirty;
  $("validate-button").disabled = state.busy;
  $("materialize-button").hidden = !state.document || !state.document.fields.some((field) => [field.value].flat().some((value) => isRemote(value) || isRemote(value?.artifact)));
  $("materialize-button").disabled = state.busy || Boolean(state.artifactTask);
  $("duplicate-button").disabled = state.busy;
  if (state.document) $("recipe-meta").textContent = `${state.document.operation} · ${builtin ? "Certified built-in" : `Revision ${state.revision}${state.dirty ? " · Unsaved edits" : " · Saved"}`}`;
  $("publication").hidden = !state.document;
  const published = state.publication;
  const enabled = Boolean(published?.enabled);
  const publicationButton = $("publication-button");
  const publicationUnavailableReason = state.busy ? "Please wait for the current operation to finish." : state.dirty ? "Save your edits before changing this setting." : !published ? "Checking publication status" : "";
  publicationButton.setAttribute("aria-checked", String(enabled));
  publicationButton.setAttribute("aria-label", "Enable recipe in LatentSlate");
  publicationButton.disabled = Boolean(publicationUnavailableReason);
  publicationButton.title = publicationUnavailableReason || (enabled ? "Enabled in LatentSlate — click to disable" : "Disabled in LatentSlate — click to enable");
  $("publication-status").textContent = !published ? "Checking status…" : published.enabled ? "Enabled" : "Disabled";
  $("publication-detail").textContent = !published ? "" : [
    "Applies to all projects using this Engine. Refresh LatentSlate’s Engine catalog after changing this setting.",
    published.tool && !published.tool.available ? `Setup required: ${published.tool.unavailable_reason}` : "",
    state.dirty ? "Save your edits before changing this setting." : "",
  ].filter(Boolean).join(" ");
}

async function loadPublication() {
  state.publication = await api(publicationPath());
  updateToolbar();
}

function publicationPath() {
  return state.builtinKey ? `/builtins/${state.builtinKey}/publication` : `/recipes/${state.document.id}/publication`;
}

$("publication-button").addEventListener("click", () => work(async () => {
  state.publication = await api(publicationPath(), { method: "PUT", body: { enabled: !state.publication.enabled } });
  await loadLibrary();
  updateToolbar();
  notice(state.publication.enabled ? "Recipe enabled. Refresh the Engine catalog in your client to use it." : "Recipe disabled. Previously accepted jobs keep their saved revision.");
}));

function markDirty() {
  state.dirty = true;
  state.validation = null;
  updateToolbar();
  renderValidation();
}

async function loadLibrary() {
  const [operations, builtins, users] = await Promise.all([api("/operations"), api("/builtins"), api("/recipes")]);
  state.operations = new Map(operations.operations.map((operation) => [operation.key, operation]));
  state.builtins = builtins.recipes;
  state.users = await Promise.all(users.recipes.map(async (record) => ({ ...record, enabled: (await api(`/recipes/${record.document.id}/publication`)).enabled })));
  $("connection-status").textContent = "Engine connected";
  $("connection-button").classList.add("connected");
  renderLibrary();
}

function renderLibrary() {
  const node = $("recipe-library");
  const scrollTop = node.scrollTop;
  node.replaceChildren();
  for (const [title, entries, builtin] of [["Built-in recipes", state.builtins, true], ["Your recipes", state.users, false]]) {
    const group = element(builtin ? "section" : "details", {
      class: "library-group", "aria-label": title,
      ...(!builtin && !state.libraryCollapsed.users ? { open: "" } : {}),
      ontoggle: (event) => {
        if (!builtin && event.currentTarget.isConnected) state.libraryCollapsed.users = !event.currentTarget.open;
      },
    });
    group.append(element(builtin ? "h2" : "summary", { class: "library-label" }, [title, element("span", { text: entries.length })]));
    if (!entries.length) group.append(element("p", { class: "no-recipes", text: "Use + to create your first recipe, or import one." }));
    const families = new Map();
    const ordered = builtin ? [...entries].sort((a, b) => {
      const familyA = libraryFamily(a.document);
      const familyB = libraryFamily(b.document);
      return (familyNames[familyA] || familyA).localeCompare(familyNames[familyB] || familyB, undefined, { numeric: true });
    }) : entries;
    for (const item of ordered) {
      let parent = group;
      let name = item.document.name;
      if (builtin) {
        const family = libraryFamily(item.document);
        const familyName = familyNames[family] || label(family);
        if (!families.has(family)) {
          const familyGroup = element("details", {
            class: "library-family", "data-family": family,
            ...(state.familyCollapsed.get(family) === false ? { open: "" } : {}),
            ontoggle: (event) => {
              if (event.currentTarget.isConnected) state.familyCollapsed.set(family, !event.currentTarget.open);
            },
          });
          familyGroup.append(element("summary", { class: "library-label" }, [familyName]));
          families.set(family, familyGroup);
          group.append(familyGroup);
        }
        parent = families.get(family);
        if (name.startsWith(`${familyName} `)) name = name.slice(familyName.length + 1);
      }
      const selected = builtin ? state.builtinKey === item.key : !state.builtinKey && state.document?.id === item.document.id;
      const identity = builtin ? "Built-in recipe" : `Your recipe · Revision ${item.revision}`;
      const publication = item.enabled ? "Enabled in LatentSlate" : "Disabled · hidden from LatentSlate";
      parent.append(element("button", {
        class: `recipe-link${selected ? " active" : ""}`,
        "data-recipe-id": item.document.id,
        "data-origin": builtin ? "builtin" : "user",
        "data-enabled": String(Boolean(item.enabled)),
        "aria-label": `${identity}: ${item.document.name}. ${publication}.`,
        title: `${item.document.name} · ${publication}`,
        ...(selected ? { "aria-current": "true" } : {}),
        ...(builtin ? { "data-builtin-key": item.key } : {}),
        onclick: () => work(async () => {
          if (state.dirty && !confirm("Discard unsaved edits and open another recipe?")) return;
          const record = builtin ? item : await api(`/recipes/${item.document.id}`);
          selectRecipe(record, builtin ? item.key : null);
          await validate(false);
        }),
      }, [
        element("span", { class: "recipe-name", text: name }),
        ...(!builtin ? [element("span", { class: "recipe-revision", text: `r${item.revision}`, "aria-hidden": "true" })] : []),
      ]));
    }
    node.append(group);
  }
  node.scrollTop = scrollTop;
}

function selectRecipe(record, builtinKey = null) {
  state.document = structuredClone(record.document);
  state.builtinKey = builtinKey;
  state.revision = record.revision ?? null;
  state.hash = record.definition_hash ?? null;
  state.dirty = false;
  state.validation = null;
  state.publication = null;
  if (builtinKey) state.familyCollapsed.set(libraryFamily(record.document), false);
  notice();
  renderLibrary();
  renderEditor();
}

async function validate(openIssues = true) {
  state.validation = await api("/validate", { method: "POST", body: state.document });
  await loadPublication();
  renderValidation(openIssues);
  return state.validation;
}

async function save() {
  await work(async () => {
    const result = await validate(true);
    if (!result.document_valid || !result.recipe_compiles) {
      notice("Fix the validation issues below before saving.", true);
      return;
    }
    const record = await api(`/recipes/${state.document.id}`, { method: "PUT", body: { base_revision: state.revision, document: state.document } });
    const validation = state.validation;
    selectRecipe(record);
    await loadPublication();
    state.validation = validation;
    renderValidation();
    await loadLibrary();
    notice(`Saved revision ${record.revision}. ${state.publication?.enabled ? "Enabled in LatentSlate; refresh its Engine catalog to pick up changes." : "Still disabled. Enable this recipe above to use it in LatentSlate."}`);
  });
}

async function reloadLatest() {
  await work(async () => {
    if (state.dirty && !confirm("Discard unsaved edits and reload the latest saved revision?")) return;
    selectRecipe(await api(`/recipes/${state.document.id}`));
    await loadLibrary();
    await validate(false);
  });
}

async function exportRecipe() {
  await work(async () => {
    const content = await api(`/recipes/${state.document.id}/export`, {}, true);
    const url = URL.createObjectURL(new Blob([content], { type: "application/json" }));
    const link = element("a", { href: url, download: `recipe-${state.document.id}.json` });
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
}

async function deleteRecipe() {
  if (state.busy || state.builtinKey || !state.document) return;
  const { id, name } = state.document;
  if (!confirm(`Permanently delete “${name}”?\n\nThis removes all local revision history and unpublishes the tool.${state.dirty ? " Your unsaved edits will also be discarded." : ""}\n\nModel files and existing generated versions are kept. Already accepted jobs can still finish.\n\nThis cannot be undone.`)) return;
  await work(async () => {
    await api(`/recipes/${id}`, { method: "DELETE" });
    state.users = state.users.filter((record) => record.document.id !== id);
    state.document = null;
    state.builtinKey = null;
    state.revision = null;
    state.hash = null;
    state.dirty = false;
    state.validation = null;
    state.publication = null;
    renderLibrary();
    renderEditor();
    await loadLibrary();
    notice("Recipe deleted. Model files and generated versions are unchanged.");
  });
}

function issueList(issues) {
  const list = element("ul", { class: "issue-list" });
  for (const issue of issues) list.append(element("li", {}, [
    element("code", { text: `${issue.stage} · ${issue.path} · ${issue.code}` }),
    issue.message, element("small", { text: issue.remediation }),
  ]));
  return list;
}

function renderImports() {
  $("import-files").disabled = state.importBusy;
  $("import-summary").textContent = state.importBusy ? "Checking recipe files…" : `${state.imports.length} files staged. Confirm each import below.`;
  const list = $("import-list");
  list.replaceChildren();
  for (const item of state.imports) {
    const card = element("section", { class: "import-card", "data-import-file": item.name });
    const preview = item.preview;
    const doc = preview?.document;
    card.append(element("span", { class: "eyebrow", text: item.name }));
    card.append(element("h3", { text: doc?.name || "Recipe file" }));
    if (doc) card.append(element("p", { class: "muted path-text", text: `${doc.operation} · ${doc.id}` }));
    if (preview) {
      const validation = preview.validation;
      const labels = { new: "New recipe", identical: "Already present · No changes", conflict: "UUID exists · Copy required", builtin: "Reserved built-in ID · Copy required", invalid: "Cannot import" };
      const statuses = element("div", { class: "import-statuses" }, [
        pill(labels[preview.status], ["conflict", "builtin", "invalid"].includes(preview.status) ? "warning" : ""),
        pill(validation.recipe_compiles ? "Policy valid" : "Policy invalid", validation.recipe_compiles ? "" : "error"),
        pill(validation.artifact_resolution.status === "resolved" ? "Dependencies resolved" : "Unresolved artifacts", validation.artifact_resolution.status === "resolved" ? "" : "warning"),
      ]);
      card.append(statuses);
      if (validation.issues.length) {
        const details = element("details", { class: "import-issues", ...(!validation.recipe_compiles ? { open: "" } : {}) }, [element("summary", { text: `${validation.issues.length} validation ${validation.issues.length === 1 ? "issue" : "issues"}` }), issueList(validation.issues)]);
        card.append(details);
      }
      if (validation.recipe_compiles && validation.artifact_resolution.status !== "resolved") card.append(element("p", { class: "field-footnote", text: "You can import this definition with unresolved paths. Artifact files are not copied." }));
      if (item.result) {
        const record = item.result.record;
        card.append(element("p", { class: "import-result", text: record ? `${item.result.status === "copied" ? "Imported as copy" : "Imported"} · Revision ${record.revision} · ${record.document.id}` : "Already present. No revision was written." }));
      } else if (["new", "conflict", "builtin"].includes(preview.status)) {
        const copy = preview.status !== "new";
        card.append(element("button", {
          class: "primary", text: copy ? "Import as copy" : "Import recipe",
          "aria-label": copy ? `Import ${item.name} as copy` : `Import ${item.name}`,
          disabled: state.importBusy, onclick: () => commitImport(item, copy),
        }));
      }
    }
    if (item.error) card.append(element("p", { class: "error-text", text: item.error }));
    if (!preview && !item.error) card.append(element("p", { class: "muted", text: "Reading and validating…" }));
    list.append(card);
  }
}

async function previewImportFiles() {
  if (state.importBusy) return;
  const files = Array.from($("import-files").files);
  state.importBusy = true;
  state.imports = files.map((file) => ({ name: file.name }));
  renderImports();
  try {
    for (let index = 0; index < files.length; index++) {
      const item = state.imports[index];
      try {
        item.source = await files[index].text();
        // Check JSON syntax locally but forward the original numeric literals.
        // Re-serializing in JavaScript changes 1.0 to 1 and the canonical hash.
        parseJSON(item.source);
        item.preview = await api("/imports/preview", { method: "POST", body: `{"document":${item.source}}` });
      } catch (error) {
        item.error = error instanceof SyntaxError ? `Invalid JSON: ${error.message}` : error.message;
        if (error.status === 401) showError(error);
      }
      renderImports();
    }
  } finally {
    state.importBusy = false;
    $("import-files").value = "";
    renderImports();
  }
}

async function commitImport(item, asCopy) {
  if (state.importBusy) return;
  state.importBusy = true;
  item.error = null;
  renderImports();
  try {
    item.result = await api("/imports", { method: "POST", body: `{"document":${item.source},"as_copy":${asCopy}}` });
    await loadLibrary();
  } catch (error) {
    item.error = error.message;
    if (error.status === 401) showError(error);
    if (error.status === 409) {
      try { item.preview = await api("/imports/preview", { method: "POST", body: `{"document":${item.source}}` }); }
      catch (previewError) { item.error = previewError.message; }
    }
  } finally {
    state.importBusy = false;
    renderImports();
  }
}

function numericValue(raw, type) {
  const text = raw.trim();
  if (!text) return null;
  if (type === "integer") {
    if (!/^-?\d+$/.test(text)) return raw;
    const integer = BigInt(text);
    return integer >= BigInt(Number.MIN_SAFE_INTEGER) && integer <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(integer) : integer;
  }
  const number = Number(text);
  return Number.isFinite(number) ? number : raw;
}

function scalarControl(descriptor, value, set, accessibleLabel, disabled) {
  const type = descriptor.value_type;
  if (descriptor.choices?.length) {
    const control = element("select", { class: "value-control", "aria-label": accessibleLabel, disabled });
    descriptor.choices.forEach((choice, index) => control.append(element("option", { value: index, text: String(choice) })));
    const selected = descriptor.choices.findIndex((choice) => stringifyJSON(choice) === stringifyJSON(value));
    control.value = selected < 0 ? "" : String(selected);
    control.addEventListener("change", () => set(descriptor.choices[Number(control.value)]));
    return control;
  }
  if (type === "boolean") {
    const checkbox = element("input", { type: "checkbox", "aria-label": accessibleLabel, checked: value === true, disabled });
    checkbox.addEventListener("change", () => set(checkbox.checked));
    return element("label", { class: "boolean-control" }, [checkbox, "Enabled"]);
  }
  if (type === "text") {
    const control = element("textarea", { "aria-label": accessibleLabel, value: value ?? "", disabled });
    control.addEventListener("input", () => set(control.value));
    return control;
  }
  const numeric = ["number", "integer"].includes(type);
  const control = element("input", {
    type: "text", class: `value-control${numeric ? " numeric-input" : ""}`,
    "aria-label": accessibleLabel, value: value === null || value === undefined ? "" : String(value),
    ...(numeric ? { inputmode: type === "integer" ? "numeric" : "decimal" } : {}), disabled,
  });
  control.addEventListener("input", () => set(numeric ? numericValue(control.value, type) : control.value));
  return control;
}

function isRemote(reference) {
  return reference?.source === "huggingface" || reference?.source === "civitai";
}

function remoteLabel(reference) {
  return reference.source === "huggingface" ? reference.repo + " · " + reference.file : "Civitai · Version " + reference.model_version_id + " · File " + reference.file_id;
}

function pathControl(descriptor, reference, set, accessibleLabel, disabled) {
  const input = element("input", { type: "text", class: "path-input", "aria-label": accessibleLabel, value: reference?.path ?? "", placeholder: descriptor.optional ? "Optional — leave empty when not used" : "Paste an absolute local path", spellcheck: "false", disabled });
  input.addEventListener("input", () => set(descriptor.optional && !input.value.trim() ? null : { source: "local", path: input.value }));
  const button = element("button", {
    class: "secondary", text: "Find", "aria-label": `Find ${accessibleLabel}`, disabled,
    onclick: () => openPicker(descriptor, (artifact) => { set(artifact); renderEditor(); }),
  });
  const sourceButton = element("button", {
    class: "secondary", text: "Hugging Face", "aria-label": "Hugging Face source for " + accessibleLabel, disabled,
    onclick: () => openHfSource(reference, (artifact) => { set(artifact); renderEditor(); }),
  });
  const civitaiButton = element("button", {
    class: "secondary", text: "Civitai", "aria-label": "Civitai source for " + accessibleLabel, disabled,
    onclick: () => openCivitaiSource(reference, (artifact) => { set(artifact); renderEditor(); }),
  });
  if (isRemote(reference)) {
    return element("div", { class: "hf-reference" }, [
      element("div", { class: "source-details" }, [
      element("strong", { text: remoteLabel(reference) }),
      element("p", { class: "hf-identity", text: (reference.source === "huggingface" ? "Pinned " + reference.revision.slice(0, 12) + " · " : "") + "SHA-256 " + reference.sha256.slice(0, 16) + "…", title: (reference.source === "huggingface" ? "Commit " + reference.revision + "\n" : "") + "SHA-256 " + reference.sha256 }),
      element("p", { class: "hf-availability", "data-sha256": reference.sha256, text: "Validate to check availability on this host" }),
      ]),
      element("div", { class: "source-actions" }, [
        element("button", { class: "quiet", text: "Local path", "aria-label": "Use local path for " + accessibleLabel, disabled, onclick: () => { set(descriptor.optional ? null : { source: "local", path: "" }); renderEditor(); } }),
        button, sourceButton, civitaiButton,
      ]),
    ]);
  }
  return element("div", { class: "input-action" }, [input, button, ...(descriptor.artifact?.kind === "file" ? [sourceButton, civitaiButton] : [])]);
}

function valueControl(descriptor, value, set, accessibleLabel, disabled) {
  if (descriptor.value_type === "artifact") return pathControl(descriptor, value, set, `${accessibleLabel} path`, disabled);
  if (descriptor.value_type === "adapter") {
    let current = value?.artifact ? value : { artifact: value ?? { source: "local", path: "" }, strength: 1 };
    const row = element("div");
    row.append(pathControl(descriptor, current.artifact, (artifact) => { current = { ...current, artifact }; set(current); }, `${accessibleLabel} path`, disabled));
    row.append(element("div", { class: "adapter-strength" }, [
      element("label", { text: "Strength" }),
      scalarControl({ value_type: "number" }, current.strength, (strength) => { current = { ...current, strength }; set(current); }, `${accessibleLabel} strength`, disabled),
    ]));
    return row;
  }
  return scalarControl(descriptor, value, set, `${accessibleLabel} value`, disabled);
}

function collectionControl(members, disabled) {
  const box = element("div", { class: "collection" });
  const { descriptor, field } = members[0];
  const values = Array.isArray(field.value) ? field.value : [];
  const title = fieldLabel(descriptor);
  if (!values.length) box.append(element("p", { class: "empty-collection", text: "No items selected." }));
  values.forEach((_, index) => {
    const row = element("div", { class: "collection-row" });
    const itemName = descriptor.presentation?.item_label ? `${descriptor.presentation.item_label} ${index + 1}` : `${title} ${index + 1}`;
    for (const [position, member] of members.entries()) {
      const memberLabel = member.descriptor.presentation?.item_label || fieldLabel(member.descriptor);
      const control = valueControl(member.descriptor, member.field.value[index], (next) => { member.field.value[index] = next; markDirty(); }, position ? `${itemName} ${memberLabel}` : itemName, disabled);
      if (!position) row.append(control);
      else row.append(element("label", { class: "collection-member" }, [
        element("span", { text: member.field.mode === "exposed" ? `Default ${memberLabel.toLowerCase()}` : memberLabel }), control,
      ]));
    }
    const move = (offset) => {
      moveCollectionItem(members, index, offset);
      markDirty(); renderEditor();
    };
    row.append(element("div", { class: "collection-controls" }, [
      element("span", { class: "row-index", text: descriptor.presentation?.item_label ? itemName : `ITEM ${String(index + 1).padStart(2, "0")}` }),
      element("button", { class: "quiet", text: "↑", "aria-label": `Move ${itemName} up`, disabled: disabled || index === 0, onclick: () => move(-1) }),
      element("button", { class: "quiet", text: "↓", "aria-label": `Move ${itemName} down`, disabled: disabled || index === values.length - 1, onclick: () => move(1) }),
      element("button", { class: "quiet", text: "Remove", "aria-label": `Remove ${itemName}`, disabled, onclick: () => { removeCollectionItem(members, index); markDirty(); renderEditor(); } }),
    ]));
    box.append(row);
  });
  box.append(element("button", {
    class: "collection-add", text: "+ Add item", "aria-label": `Add ${title} item`, disabled,
    onclick: () => { appendCollectionItem(members); markDirty(); renderEditor(); },
  }));
  return box;
}

function fieldModeControl(descriptor, field, disabled) {
  const mode = element("select", { class: "mode-select", "aria-label": `${fieldLabel(descriptor)} mode`, disabled }, [element("option", { value: "fixed", text: "Fixed" }), element("option", { value: "exposed", text: "Exposed" })]);
  mode.value = field.mode;
  mode.addEventListener("change", () => { field.mode = mode.value; markDirty(); renderEditor(); });
  return mode;
}

function constraintsControl(descriptor, field, disabled, summary = "Narrowed constraints") {
  const details = element("details", { class: "constraint-details" }, [element("summary", { text: summary })]);
  const grid = element("div", { class: "constraint-grid" });
  if (["number", "integer"].includes(descriptor.value_type)) {
    for (const key of ["minimum", "maximum", "step"]) {
      const name = `${fieldLabel(descriptor)} ${key}`;
      const input = element("input", { "aria-label": name, value: field[key] === undefined ? "" : String(field[key]), placeholder: descriptor[key] === null || descriptor[key] === undefined ? "Inherit" : `Inherit ${descriptor[key]}`, inputmode: "decimal", disabled });
      input.addEventListener("input", () => {
        if (!input.value.trim()) delete field[key];
        else field[key] = numericValue(input.value, descriptor.value_type);
        markDirty();
      });
      grid.append(element("label", {}, [element("span", { class: "control-label", text: label(key) }), input]));
    }
  }
  const choices = element("input", { "aria-label": `${fieldLabel(descriptor)} choices`, value: field.choices === undefined ? "" : stringifyJSON(field.choices), placeholder: "Inherit · or a JSON array", disabled });
  choices.addEventListener("input", () => {
    if (!choices.value.trim()) delete field.choices;
    else { try { field.choices = parseJSON(choices.value); } catch { field.choices = choices.value; } }
    markDirty();
  });
  grid.append(element("label", { class: "full-width" }, [element("span", { class: "control-label", text: "Choices" }), choices]));
  if (descriptor.optional) {
    const nullable = element("select", { "aria-label": `${fieldLabel(descriptor)} nullability`, disabled }, [
      element("option", { value: "inherit", text: "Inherit family nullability" }),
      element("option", { value: "true", text: "Allow an empty value" }),
      element("option", { value: "false", text: "Require a value" }),
    ]);
    nullable.value = field.nullable === undefined ? "inherit" : String(field.nullable);
    nullable.addEventListener("change", () => {
      if (nullable.value === "inherit") delete field.nullable;
      else field.nullable = nullable.value === "true";
      markDirty();
    });
    grid.append(element("label", { class: "full-width" }, [element("span", { class: "control-label", text: "Empty values" }), nullable]));
  }
  details.append(grid);
  return details;
}

function renderEditor() {
  $("empty-state").hidden = Boolean(state.document);
  $("editor").hidden = !state.document;
  if (!state.document) return;
  const disabled = Boolean(state.builtinKey);
  $("recipe-kind").textContent = disabled ? "BUILT-IN RECIPE" : "USER RECIPE";
  $("recipe-name").value = state.document.name;
  $("recipe-name").readOnly = disabled;
  $("recipe-name-label").hidden = disabled;
  $("readonly-banner").hidden = !disabled;
  const sources = $("builtin-sources");
  sources.hidden = !disabled;
  sources.replaceChildren();
  if (disabled) {
    const assets = state.builtins.find((item) => item.key === state.builtinKey)?.bootstrap_assets || [];
    const details = element("details", { class: "field-card" }, [
      element("summary", { text: "Review built-in family sources" }),
      element("p", { class: "field-footnote", text: "Pinned download sources for this family's bootstrap. The artifact fields below show the files this recipe currently uses." }),
    ]);
    for (const asset of assets) {
      const ref = asset.reference;
      const url = ref.source === "huggingface"
        ? `https://huggingface.co/${ref.repo}/blob/${ref.revision}/${ref.file.split("/").map(encodeURIComponent).join("/")}`
        : ref.url;
      details.append(element("p", { class: "field-footnote" }, [
        element("a", { href: url, target: "_blank", rel: "noopener noreferrer", text: asset.path, title: url }),
        element("span", { text: ` · ${ref.source === "huggingface" ? "Hugging Face" : "Tokenizer support"} · ${(asset.size_bytes / 1024 ** 2).toFixed(1)} MiB` }),
        element("br"),
        element("span", { text: `SHA-256 ${ref.sha256}`, style: "overflow-wrap:anywhere" }),
      ]));
    }
    sources.append(details);
  }
  $("artifact-fields").replaceChildren();
  $("policy-fields").replaceChildren();
  $("caller-fields").replaceChildren();
  const operation = state.operations.get(state.document.operation);
  if (state.document.operation.startsWith("ltx23.") && !state.document.fields.some((field) => field.key === "fps")) {
    state.document.fields.push({ key: "fps", mode: "fixed", value: 30 });
  }
  const descriptors = new Map(operation.fields.map((field) => [field.key, field]));
  if (state.document.operation === "krea2.t2i" && !state.document.fields.some((field) => field.key === "prompt_enhancement")) {
    state.document.fields.push({ key: "prompt_enhancement", mode: "exposed", value: false });
  }
  const fields = new Map(state.document.fields.map((field) => [field.key, field]));
  const groups = new Map((operation.field_groups || [])
    .filter((group) => group.layout === "collection" && group.fields.every((key) => fields.has(key)))
    .flatMap((group) => group.fields.map((key) => [key, group])));
  for (const field of state.document.fields) {
    const descriptor = descriptors.get(field.key);
    if (!descriptor || descriptor.owner === "host") continue;
    const group = groups.get(field.key);
    if (group && field.key !== group.fields[0]) continue;
    const members = (group?.fields || [field.key]).map((key) => ({ descriptor: descriptors.get(key), field: fields.get(key) }));
    if (descriptor.owner === "caller") {
      $("caller-fields").append(element("div", { class: "caller-chip" }, [label(field.key), element("span", { text: descriptor.value_type })]));
      continue;
    }
    const artifact = descriptor.owner === "artifact";
    const card = element("div", { class: `field-card${descriptor.ordered || descriptor.value_type === "text" ? " wide-field" : ""}`, "data-field": field.key });
    const heading = element("div", { class: "field-heading" }, [element("h3", { text: fieldLabel(descriptor) })]);
    if (artifact) heading.append(element("span", { class: "slot-status", "data-slot": field.key, text: "Not checked" }));
    else heading.append(fieldModeControl(descriptor, field, disabled));
    card.append(heading);
    const groupedPolicy = group ? members.slice(1).filter((member) => member.descriptor.owner === "recipe") : [];
    for (const member of groupedPolicy) {
      card.append(element("label", { class: "collection-policy" }, [
        member.descriptor.presentation?.item_label || fieldLabel(member.descriptor), fieldModeControl(member.descriptor, member.field, disabled),
      ]));
    }
    const presentation = descriptor.presentation;
    const warning = presentation?.advanced_warning ? element("p", { class: "field-footnote", text: presentation.advanced_warning }) : null;
    const updateWarning = () => {
      if (warning) warning.hidden = field.mode === "fixed" && field.value === presentation.certified_value;
    };
    if (!artifact) card.append(element("span", { class: "control-label", text: field.mode === "fixed" ? "Fixed value" : "Default value" }));
    if (descriptor.ordered) card.append(collectionControl(members, disabled));
    else card.append(valueControl(descriptor, field.value, (value) => { field.value = value; markDirty(); updateWarning(); }, fieldLabel(descriptor), disabled));
    for (const member of groupedPolicy) {
      card.append(constraintsControl(member.descriptor, member.field, disabled, `${member.descriptor.presentation?.item_label || fieldLabel(member.descriptor)} constraints`));
    }
    if (warning) { updateWarning(); card.append(warning); }
    if (artifact) {
      const companions = descriptor.artifact.required_files;
      if (companions?.length) card.append(element("p", { class: "field-footnote", text: `Directory requires ${companions.length} companion files. Validate to check the selected path.` }));
    } else {
      if (descriptor.choices?.length) card.append(element("p", { class: "field-footnote", text: `Family supports: ${descriptor.choices.join(", ")}` }));
      card.append(constraintsControl(descriptor, field, disabled));
    }
    $(artifact ? "artifact-fields" : "policy-fields").append(card);
  }
  updateToolbar();
  renderValidation();
}

function renderValidation(openIssues = false) {
  document.querySelectorAll(".hf-availability").forEach((node) => {
    const slots = state.validation?.artifact_resolution.slots.filter((slot) => slot.reference?.sha256 === node.dataset.sha256);
    node.textContent = !slots?.length ? "Validate to check availability on this host" : slots.every((slot) => slot.status === "resolved") ? "Available on this host" : "Not materialized on this host";
  });
  const summary = $("validation-summary");
  const details = $("validation-details");
  summary.replaceChildren(); details.replaceChildren();
  const result = state.validation;
  if (!result) summary.append(pill(state.dirty ? "Unvalidated edits" : "Not validated", "neutral"), element("span", { text: "Validate to check policy and local paths." }));
  else {
    summary.append(pill(result.recipe_compiles ? "Policy valid" : "Policy needs attention", result.recipe_compiles ? "" : "error"));
    summary.append(pill(result.artifact_resolution.status === "resolved" ? "Dependencies resolved" : "Unresolved artifacts", result.artifact_resolution.status === "resolved" ? "" : "warning"));
    summary.append(element("span", { text: "Execution unverified · Backend not checked" }));
    if (result.execution_readiness.status === "blocked") summary.lastChild.textContent = "Execution blocked · Backend not checked";
    if (result.issues.length) {
      const panel = element("details", { ...(openIssues ? { open: "" } : {}) }, [element("summary", { text: `${result.issues.length} validation ${result.issues.length === 1 ? "issue" : "issues"}` })]);
      panel.append(issueList(result.issues)); details.append(panel);
    }
  }
  document.querySelectorAll("[data-slot]").forEach((node) => {
    const slots = result?.artifact_resolution.slots.filter((slot) => slot.key === node.dataset.slot);
    const resolved = slots && slots.every((slot) => slot.status === "resolved");
    node.textContent = !result ? "Not checked" : resolved ? "Resolved" : "Unresolved";
    node.className = `slot-status${result ? resolved ? " resolved" : " unresolved" : ""}`;
  });
}

function byteSize(value) {
  if (value === null || value === undefined) return "Unknown size";
  if (value < 1024) return value + " B";
  if (value < 1024 ** 2) return (value / 1024).toFixed(1) + " KB";
  if (value < 1024 ** 3) return (value / 1024 ** 2).toFixed(1) + " MB";
  return (value / 1024 ** 3).toFixed(2) + " GB";
}

function artifactTaskControls(dialog, active) {
  if (active && !state.artifactTask) {
    state.artifactTask = { id: null, status: "starting" };
    updateToolbar();
  }
  if (!active) {
    state.artifactTask = null;
    updateToolbar();
  }
  dialog.querySelectorAll("input,select,button").forEach((node) => {
    if (!node.hasAttribute("data-cancel-artifact")) node.disabled = active;
  });
  dialog.querySelector("[data-cancel-artifact]").hidden = !active;
  if (dialog.id === "materialization-dialog" && !active) $("start-materialization").disabled = !state.materializationPlan?.summary.missing;
}

async function followArtifactTask(task, statusNode) {
  state.artifactTask = task;
  updateToolbar();
  try {
    while (true) {
      statusNode.textContent = task.stage + " · " + byteSize(task.bytes_downloaded) + (task.total_bytes === null ? "" : " / " + byteSize(task.total_bytes));
      if (task.plan) renderMaterializationPlan(task.plan);
      if (task.status !== "running") {
        if (task.status === "failed") throw new Error(task.error || "Artifact task failed");
        if (task.status === "canceled") throw new Error("Canceled. Completed cache files are kept; partial files are not published.");
        return task.result;
      }
      await new Promise((resolve) => setTimeout(resolve, 750));
      task = await api("/materializations/" + task.id);
      state.artifactTask = task;
    }
  } finally {
    state.artifactTask = null;
    updateToolbar();
  }
}

async function openHfSource(reference, select) {
  if (state.artifactTask) return;
  state.hfPicker = select;
  $("hf-current-source").textContent = "Current: " + (isRemote(reference) ? remoteLabel(reference) : reference?.path || "No file selected");
  $("use-hf-source").textContent = reference?.source === "local" && reference.path ? "Replace local reference" : isRemote(reference) ? "Replace online reference" : "Use this file";
  state.pinnedSource = null;
  $("hf-source-form").reset();
  $("hf-source-url").value = "";
  $("hf-source-repo").value = reference?.repo || "";
  $("hf-source-file").value = reference?.file || "";
  $("hf-source-revision").value = reference?.revision || "main";
  $("hf-source-status").textContent = "";
  $("hf-source-error").textContent = "";
  $("hf-pinned-preview").textContent = "";
  $("use-hf-source").hidden = true;
  $("hf-source-dialog").showModal();
  $("hf-source-url").focus();
  try {
    const source = await api("/sources/huggingface");
    $("hf-auth-status").textContent = source.authentication_configured ? "Host authentication configured." : "No host authentication configured. Public files work without a token.";
  } catch (error) { $("hf-auth-status").textContent = error.message; }
}

async function pinHfSource(event) {
  event.preventDefault();
  if (state.artifactTask) return;
  const url = $("hf-source-url").value.trim();
  const locator = url ? { url } : { repo: $("hf-source-repo").value.trim(), file: $("hf-source-file").value.trim(), revision: $("hf-source-revision").value.trim() || "main" };
  const dialog = $("hf-source-dialog");
  $("hf-source-error").textContent = "";
  $("hf-pinned-preview").textContent = "";
  $("use-hf-source").hidden = true;
  state.pinnedSource = null;
  artifactTaskControls(dialog, true);
  try {
    const task = await api("/sources/huggingface/pin", { method: "POST", body: locator });
    const result = await followArtifactTask(task, $("hf-source-status"));
    state.pinnedSource = result.reference;
    $("hf-pinned-preview").textContent = "Proposed online source: " + result.reference.repo + " · " + result.reference.file + "\nCommit " + result.reference.revision + "\nSHA-256 " + result.reference.sha256 + "\n" + byteSize(result.size_bytes) + (result.downloaded_for_hash ? " · Downloaded to establish checksum" : " · Source checked; use Download missing files after applying");
    $("use-hf-source").hidden = false;
  } catch (error) { $("hf-source-error").textContent = error.message; }
  finally { artifactTaskControls(dialog, false); }
}

function renderMaterializationPlan(plan) {
  state.materializationPlan = plan;
  const summary = plan.summary;
  $("materialization-summary").textContent = summary.unique_artifacts + " unique artifacts · " + summary.cached + " cached · " + summary.resolved_local + " local · " + summary.missing + " to download" + (summary.unresolved ? " · " + summary.unresolved + " unresolved" : "") + ". Download: " + byteSize(summary.download_bytes_known) + (summary.unknown_sizes ? " + " + summary.unknown_sizes + " sizes checked at download" : "") + ".";
  $("materialization-list").replaceChildren(...plan.dependencies.map((entry) => element("div", { class: "dependency-row" }, [
    element("strong", { text: isRemote(entry.reference) ? remoteLabel(entry.reference) : entry.reference.path }),
    element("span", { class: "slot-status " + (["cached", "resolved_local"].includes(entry.status) ? "resolved" : "unresolved"), text: label(entry.status) + " · " + entry.consumers.length + " recipe slots" }),
    ...(entry.message ? [element("p", { class: "muted", text: entry.message })] : []),
  ])));
}

async function openMaterialization() {
  if (state.busy || state.artifactTask) return;
  state.materializationDocument = structuredClone(state.document);
  state.materializationPlan = null;
  $("materialization-error").textContent = "";
  $("materialization-status").textContent = "";
  $("materialization-summary").textContent = "Checking dependencies…";
  $("materialization-list").replaceChildren();
  $("start-materialization").disabled = true;
  $("materialization-dialog").showModal();
  try {
    const plan = await api("/materializations/plan", { method: "POST", body: { documents: [state.materializationDocument] } });
    renderMaterializationPlan(plan);
    $("start-materialization").disabled = !plan.summary.missing;
  } catch (error) { $("materialization-error").textContent = error.message; }
}

async function startMaterialization() {
  if (state.artifactTask) return;
  const dialog = $("materialization-dialog");
  $("materialization-error").textContent = "";
  artifactTaskControls(dialog, true);
  try {
    const task = await api("/materializations", { method: "POST", body: { documents: [state.materializationDocument] } });
    const result = await followArtifactTask(task, $("materialization-status"));
    $("materialization-status").textContent = result.resolved ? "Available on this host. The recipe definition has not changed." : "Downloads complete. Some local dependencies still need attention.";
    await work(() => validate(false));
    renderMaterializationPlan(await api("/materializations/plan", { method: "POST", body: { documents: [state.materializationDocument] } }));
  } catch (error) { $("materialization-error").textContent = error.message; }
  finally { artifactTaskControls(dialog, false); }
}

$("hf-source-form").addEventListener("submit", pinHfSource);
$("hf-source-form").addEventListener("input", () => {
  state.pinnedSource = null;
  $("use-hf-source").hidden = true;
  $("hf-pinned-preview").textContent = "";
  $("hf-source-status").textContent = "";
});
$("use-hf-source").addEventListener("click", async () => {
  if (!state.pinnedSource) return;
  state.hfPicker(state.pinnedSource);
  $("hf-source-dialog").close();
  await work(() => validate(false));
});
$("materialize-button").addEventListener("click", openMaterialization);
$("start-materialization").addEventListener("click", startMaterialization);
document.querySelectorAll("[data-cancel-artifact]").forEach((button) => button.addEventListener("click", async () => {
  if (state.artifactTask?.id) {
    try { await api("/materializations/" + state.artifactTask.id, { method: "DELETE" }); }
    catch (error) { button.closest("dialog").querySelector(".error-text").textContent = error.message; }
  }
}));
for (const id of ["hf-source-dialog", "civitai-source-dialog", "materialization-dialog", "library-downloads-dialog"]) {
  $(id).addEventListener("cancel", (event) => { if (state.artifactTask) event.preventDefault(); });
}

function clearCivitaiPin() {
  state.civitaiPinned = null;
  $("use-civitai-source").hidden = true;
  $("civitai-pinned-preview").textContent = "";
  $("civitai-source-status").textContent = "";
}

async function openCivitaiSource(reference, select) {
  if (state.artifactTask) return;
  state.civitaiPicker = select;
  $("civitai-current-source").textContent = "Current: " + (isRemote(reference) ? remoteLabel(reference) : reference?.path || "No file selected");
  $("use-civitai-source").textContent = reference?.source === "local" && reference.path ? "Replace local reference" : isRemote(reference) ? "Replace online reference" : "Use this file";
  state.civitaiVersion = null;
  clearCivitaiPin();
  $("civitai-source-form").reset();
  $("civitai-locator").value = reference?.source === "civitai" ? reference.model_version_id : "";
  $("civitai-version-name").textContent = "";
  $("civitai-files").replaceChildren();
  $("civitai-file-details").textContent = "";
  $("civitai-file-selection").hidden = true;
  $("pin-civitai-source").disabled = true;
  $("civitai-source-error").textContent = "";
  $("civitai-source-dialog").showModal();
  $("civitai-locator").focus();
  try {
    const source = await api("/sources/civitai");
    $("civitai-auth-status").textContent = source.authentication_configured ? "Host authentication configured." : "No host authentication configured. Public files work where Civitai permits anonymous downloads.";
  } catch (error) { $("civitai-auth-status").textContent = error.message; }
}

function selectCivitaiFile() {
  clearCivitaiPin();
  const file = state.civitaiVersion?.files.find((file) => String(file.file_id) === $("civitai-files").value);
  $("pin-civitai-source").disabled = !file;
  $("civitai-file-details").textContent = file ? ["File " + file.file_id, file.type, file.format, (file.size_bytes_estimate === null ? "Unknown size" : "About " + byteSize(file.size_bytes_estimate)), file.primary ? "Primary" : "", file.sha256 ? "SHA-256 " + file.sha256 : "No SHA-256 reported; checking downloads the file to calculate it."].filter(Boolean).join(" · ") : "Choose the exact online file.";
}

$("civitai-source-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (state.artifactTask) return;
  clearCivitaiPin();
  state.civitaiVersion = null;
  $("civitai-file-selection").hidden = true;
  $("civitai-source-error").textContent = "";
  const input = $("civitai-locator").value.trim();
  const locator = /^[1-9][0-9]*$/.test(input) ? { model_version_id: parseJSON(input) } : { url: input };
  const dialog = $("civitai-source-dialog");
  artifactTaskControls(dialog, true);
  dialog.querySelector("[data-cancel-artifact]").hidden = true;
  try {
    const version = await api("/sources/civitai/inspect", { method: "POST", body: locator });
    state.civitaiVersion = version;
    $("civitai-version-name").textContent = version.model_name + " · " + version.version_name + " · Version " + version.model_version_id;
    $("civitai-files").replaceChildren(element("option", { value: "", text: "Choose a file…" }), ...version.files.map((file) => element("option", { value: file.file_id, text: file.name + " · File " + file.file_id + (file.primary ? " · Primary" : "") })));
    const primary = version.files.filter((file) => file.primary);
    const selected = version.selected_file_id != null ? version.files.find((file) => file.file_id === version.selected_file_id) : version.files.length === 1 ? version.files[0] : primary.length === 1 ? primary[0] : null;
    $("civitai-files").value = selected ? String(selected.file_id) : "";
    $("civitai-file-selection").hidden = false;
  } catch (error) { $("civitai-source-error").textContent = error.message; }
  finally { artifactTaskControls(dialog, false); selectCivitaiFile(); }
});

$("civitai-files").addEventListener("change", selectCivitaiFile);
$("civitai-locator").addEventListener("input", () => {
  state.civitaiVersion = null;
  $("civitai-file-selection").hidden = true;
  clearCivitaiPin();
  $("pin-civitai-source").disabled = true;
});
$("pin-civitai-source").addEventListener("click", async () => {
  const file = state.civitaiVersion?.files.find((file) => String(file.file_id) === $("civitai-files").value);
  if (!file || state.artifactTask) return;
  clearCivitaiPin();
  $("civitai-source-error").textContent = "";
  const dialog = $("civitai-source-dialog");
  artifactTaskControls(dialog, true);
  try {
    const task = await api("/sources/civitai/pin", { method: "POST", body: { model_version_id: state.civitaiVersion.model_version_id, file_id: file.file_id } });
    const result = await followArtifactTask(task, $("civitai-source-status"));
    state.civitaiPinned = result.reference;
    $("civitai-pinned-preview").textContent = "Proposed online source: " + remoteLabel(result.reference) + "\nSHA-256 " + result.reference.sha256 + (result.downloaded_for_hash ? "\nDownloaded to establish checksum" : "\nSource checked; use Download missing files after applying");
    $("use-civitai-source").hidden = false;
  } catch (error) { $("civitai-source-error").textContent = error.message; }
  finally { artifactTaskControls(dialog, false); }
});
$("use-civitai-source").addEventListener("click", async () => {
  if (!state.civitaiPinned) return;
  state.civitaiPicker(state.civitaiPinned);
  $("civitai-source-dialog").close();
  await work(() => validate(false));
});

function downloadSelection() {
  const checked = state.downloadRecipes.filter((item) => item.checked);
  return { builtin_keys: checked.filter((item) => item.key).map((item) => item.key), recipe_ids: checked.filter((item) => !item.key).map((item) => item.document.id) };
}

function renderDownloadRecipes() {
  $("download-recipes").replaceChildren(...state.downloadRecipes.map((item) => element("label", { class: "download-recipe" }, [
    element("input", { type: "checkbox", checked: item.checked, onchange: (event) => { item.checked = event.target.checked; $("library-download-result").textContent = ""; $("library-download-progress").hidden = true; refreshDownloadPreview(); } }),
    element("span", { text: item.document.name + (item.key ? " · Built-in" : "") }),
  ])));
}

function renderDownloadPlan(plan) {
  state.downloadPlan = plan;
  const s = plan.summary;
  $("library-download-summary").textContent = `${s.recipes} recipes · ${s.missing} files to download · ${byteSize(s.download_bytes_known)}${s.unknown_sizes ? ` + ${s.unknown_sizes} unknown sizes` : ""} · ${s.available} available${s.install ? ` · ${s.install} to link from existing files` : ""}${s.unresolved ? ` · ${s.unresolved} need attention` : ""}`;
  const labels = { needs_download: "To download", needs_install: "Link existing file", cached: "Available", resolved_local: "Available", unresolved: "Needs attention", downloading: "Downloading", failed: "Not completed" };
  const order = { failed: 0, downloading: 1, needs_download: 2, needs_install: 3, unresolved: 4, cached: 5, resolved_local: 5 };
  const expanded = new Set([...$("library-download-files").querySelectorAll("details[open]")].map((node) => node.dataset.downloadId));
  $("library-download-files").replaceChildren(...[...plan.dependencies].sort((a, b) => order[a.status] - order[b.status]).map((entry) => {
    const ref = entry.reference;
    const name = ref.source === "builtin_support" ? ref.url.split("/").at(-1) : isRemote(ref) ? remoteLabel(ref) : ref.path;
    const card = element("details", { class: "download-file", "data-download-id": entry.id }, [
      element("summary", { text: `${labels[entry.status] || entry.status} · ${name}` }),
      element("p", { class: "field-footnote", text: `${ref.source === "builtin_support" ? "Official tokenizer support" : label(ref.source)} · ${entry.size_bytes == null ? "Size unknown" : byteSize(entry.size_bytes)}` }),
      element("p", { class: "field-footnote", text: "Used by: " + [...new Set(entry.consumers.map((consumer) => consumer.name))].join(", ") }),
    ]);
    card.open = expanded.has(entry.id);
    if (entry.message) card.append(element("p", { class: "field-footnote", text: entry.message }));
    if (entry.status === "unresolved") card.append(element("button", {
      class: "quiet", text: ref.source === "local" ? "Locate file in recipe" : "Review recipe", disabled: Boolean(state.artifactTask),
      onclick: () => {
        const item = state.downloadRecipes.find((item) => item.document.id === entry.consumers[0].id);
        if (!item || state.dirty && !confirm("Discard unsaved edits and open another recipe?")) return;
        $("library-downloads-dialog").close();
        selectRecipe(item, item.key || null);
        document.querySelector(`[data-field="${entry.consumers[0].field}"]`)?.scrollIntoView({ block: "center" });
      },
    }));
    return card;
  }));
}

async function refreshDownloadPreview() {
  const sequence = ++state.downloadSequence;
  state.downloadPlan = null;
  $("library-download-start").disabled = true;
  $("library-download-summary").textContent = "Checking selected recipes…";
  $("library-download-error").textContent = "";
  try {
    const plan = await api("/library-downloads/plan", { method: "POST", body: downloadSelection() });
    if (sequence !== state.downloadSequence || !$("library-downloads-dialog").open) return;
    renderDownloadPlan(plan);
    $("library-download-start").disabled = Boolean(state.artifactTask) || !(plan.summary.missing + plan.summary.install);
  } catch (error) {
    if (sequence === state.downloadSequence) $("library-download-error").textContent = error.message;
  }
}

$("manage-downloads-button").addEventListener("click", () => work(async () => {
  if (state.artifactTask) { notice("Wait for the current download task or cancel it first."); return; }
  await loadLibrary();
  state.downloadRecipes = [...state.builtins, ...state.users].map((item) => ({ ...item, checked: item.enabled }));
  renderDownloadRecipes();
  $("library-download-progress").hidden = true;
  $("library-download-result").textContent = "";
  $("library-download-start").textContent = "Download now";
  $("library-downloads-dialog").showModal();
  await refreshDownloadPreview();
}));
document.querySelectorAll("[data-download-selection]").forEach((button) => button.addEventListener("click", () => {
  for (const item of state.downloadRecipes) item.checked = button.dataset.downloadSelection === "all" || button.dataset.downloadSelection === "enabled" && item.enabled;
  renderDownloadRecipes();
  $("library-download-result").textContent = "";
  $("library-download-progress").hidden = true;
  refreshDownloadPreview();
}));
$("library-download-start").addEventListener("click", async () => {
  if (!state.downloadPlan || state.artifactTask) return;
  const dialog = $("library-downloads-dialog");
  const selection = downloadSelection();
  const total = state.downloadPlan.summary.download_bytes_known;
  const unknown = state.downloadPlan.summary.unknown_sizes;
  ++state.downloadSequence;
  artifactTaskControls(dialog, true);
  $("library-download-progress").hidden = false;
  $("library-download-result").textContent = "";
  $("library-download-error").textContent = "";
  let failed = false;
  try {
    let task = await api("/library-downloads", { method: "POST", body: selection });
    while (true) {
      state.artifactTask = task;
      if (task.plan) renderDownloadPlan(task.plan);
      $("library-download-stage").textContent = `${task.completed_files || 0}/${task.file_count ?? "?"} files completed · ${task.stage}`;
      $("library-download-bytes").textContent = `${byteSize(task.overall_bytes || 0)} downloaded / ${byteSize(total)}${unknown ? " + unknown sizes" : ""} · Current file: ${byteSize(task.bytes_downloaded || 0)}${task.total_bytes == null ? "" : " / " + byteSize(task.total_bytes)}`;
      const bar = $("library-download-bar");
      if (task.total_bytes > 0) { bar.max = task.total_bytes; bar.value = task.bytes_downloaded || 0; }
      else bar.removeAttribute("value");
      if (task.status !== "running") {
        if (task.status === "succeeded") $("library-download-result").textContent = `${task.result.ready} recipes ready; ${task.result.needs_attention} need attention. Recipe enablement and saved definitions are unchanged.`;
        else { failed = true; $("library-download-error").textContent = task.status === "canceled" ? "Canceled. Completed files were kept; retry downloads only what remains." : task.error; }
        break;
      }
      await new Promise((resolve) => setTimeout(resolve, 750));
      task = await api("/materializations/" + task.id);
    }
  } catch (error) { failed = true; $("library-download-error").textContent = error.message; }
  finally {
    artifactTaskControls(dialog, false);
    const errorText = $("library-download-error").textContent;
    await refreshDownloadPreview();
    $("library-download-error").textContent = errorText;
    $("library-download-start").textContent = failed ? "Retry remaining" : "Download now";
  }
});

function openConnection() {
  $("engine-address").textContent = window.location.origin;
  $("engine-token").value = state.token;
  if (!$("connection-dialog").open) $("connection-dialog").showModal();
}

async function connect(event) {
  event?.preventDefault();
  state.token = $("engine-token").value;
  if (state.token) sessionStorage.setItem("latentslate.authoring.token", state.token);
  else sessionStorage.removeItem("latentslate.authoring.token");
  $("connection-error").textContent = "";
  try {
    await loadLibrary();
    $("connection-dialog").close();
    notice();
    if (!state.document && state.builtins.length) {
      selectRecipe(state.builtins[0], state.builtins[0].key);
      await work(() => validate(false));
    }
  } catch (error) {
    $("connection-error").textContent = error.message;
    showError(error);
  }
}

async function loadRoots() {
  const { roots } = await api("/roots");
  const node = $("root-list");
  node.replaceChildren();
  if (!roots.length) node.append(element("p", { class: "muted", text: "No model folders registered. Manual recipe paths work without a model folder." }));
  for (const root of roots) {
    const remove = element("button", {
      class: "quiet", text: "Remove", "aria-label": `Remove model folder ${root.name || root.path}`,
      onclick: async () => {
        remove.disabled = true;
        try {
          await api(`/roots/${root.id}`, { method: "DELETE" });
          await loadRoots();
          $("root-index-status").textContent = "Index will refresh on the next search.";
        } catch (error) { $("root-error").textContent = error.message; }
        finally { remove.disabled = false; }
      },
    });
    node.append(element("div", { class: "root-item" }, [element("div", {}, [
      element("strong", { text: root.name || root.path.split(/[\\/]/).filter(Boolean).at(-1) || root.path }),
      element("p", { class: "path-text", text: root.path }),
      pill(root.available ? "Available" : "Unavailable · Still registered", root.available ? "" : "warning"),
    ]), remove]));
  }
}

async function openRoots() {
  $("root-error").textContent = "";
  if (!$("roots-dialog").open) $("roots-dialog").showModal();
  try { await loadRoots(); } catch (error) { $("root-error").textContent = error.message; showError(error); }
}

async function addRoot(event) {
  event.preventDefault();
  const button = $("root-form").querySelector("button[type=submit]");
  button.disabled = true; $("root-error").textContent = "";
  try {
    await api("/roots", { method: "POST", body: { path: $("root-path").value, name: $("root-name").value || null } });
    $("root-path").value = ""; $("root-name").value = "";
    await loadRoots();
    $("root-index-status").textContent = "Folder registered. Search will build its index.";
  } catch (error) { $("root-error").textContent = error.message; }
  finally { button.disabled = false; }
}

async function refreshIndex() {
  const buttons = [$("refresh-roots-button"), $("refresh-picker-button")];
  buttons.forEach((button) => { button.disabled = true; });
  $("root-index-status").textContent = "Indexing registered folders…";
  $("search-summary").textContent = "Indexing registered folders…";
  try {
    const index = await api("/artifacts/refresh", { method: "POST" });
    $("root-index-status").textContent = `${index.files} files · ${index.directories} folders indexed`;
    if (state.picker) await searchArtifacts();
  } catch (error) {
    $("root-index-status").textContent = error.message;
    $("search-summary").textContent = error.message;
    showError(error);
  } finally { buttons.forEach((button) => { button.disabled = false; }); }
}

function openPicker(descriptor, select) {
  state.picker = { descriptor, select, operation: state.document.operation, folders: [], folderSuggestions: [] };
  $("picker-title").textContent = fieldLabel(descriptor);
  $("artifact-query").value = "";
  $("search-results").replaceChildren();
  renderFolderFilters();
  $("picker-dialog").showModal();
  $("artifact-query").focus();
  searchArtifacts();
}

function setFolderFilters(folders) {
  if (!state.picker) return;
  state.picker.folders = folders;
  state.picker.folderSuggestions = [];
  clearTimeout(state.searchTimer);
  renderFolderFilters();
  $("artifact-query").focus();
  searchArtifacts();
}

function addFolderFilter(folder) {
  const selected = state.picker.folders.filter((current) => current.id !== folder.id
    && !folder.ancestors.includes(current.id) && !current.ancestors.includes(folder.id));
  setFolderFilters([...selected, folder]);
}

function renderFolderFilters() {
  const picker = state.picker;
  const selected = $("selected-folders");
  const suggested = $("suggested-folders");
  selected.replaceChildren();
  suggested.replaceChildren();
  selected.hidden = !picker.folders.length;
  const suggestions = picker.folderSuggestions.filter((folder) => !picker.folders.some((current) => current.id === folder.id));
  suggestions.sort((a, b) => Number(picker.folders.some((current) => b.ancestors.includes(current.id)))
    - Number(picker.folders.some((current) => a.ancestors.includes(current.id))));
  suggested.hidden = !suggestions.length;
  $("folder-filters").hidden = selected.hidden && suggested.hidden;
  if (picker.folders.length) {
    selected.append(element("span", { class: "folder-label", text: "In folders" }));
    for (const folder of picker.folders) {
      const path = folder.relative_path === "." ? folder.root_name : `${folder.root_name} / ${folder.relative_path.replaceAll("\\", "/")}`;
      selected.append(element("button", {
        class: "folder-chip selected", "aria-label": `Remove folder filter ${path}`,
        title: folder.absolute_path,
        onclick: () => setFolderFilters(picker.folders.filter((current) => current.id !== folder.id)),
      }, [element("span", { text: path }), element("span", { class: "folder-remove", text: "×", "aria-hidden": "true" })]));
    }
    selected.append(element("button", { class: "quiet", text: "Clear filters", onclick: () => setFolderFilters([]) }));
    selected.append(element("span", { class: "folder-hint", text: "Includes subfolders" }));
  }
  if (suggestions.length) {
    suggested.append(element("span", { class: "folder-label", text: "Folders" }));
    const roots = new Set(picker.folderSuggestions.map((folder) => folder.root_id));
    for (const folder of suggestions) {
      const relative = folder.relative_path.replaceAll("\\", "/");
      const path = relative === "." ? folder.root_name : roots.size > 1 ? `${folder.root_name} / ${relative}` : relative;
      const narrowing = picker.folders.some((current) => folder.ancestors.includes(current.id));
      suggested.append(element("button", {
        class: "folder-chip", "aria-label": `Filter by folder ${path}`,
        title: `${narrowing ? "Narrow to" : "Search in"} ${folder.absolute_path} (${folder.count} results)`,
        onclick: () => addFolderFilter(folder),
      }, [element("span", { text: path }), element("span", { class: "folder-count", text: folder.count, "aria-hidden": "true" })]));
    }
  }
}

async function searchArtifacts() {
  if (!state.picker) return;
  const sequence = ++state.searchSequence;
  const picker = state.picker;
  $("search-summary").textContent = "Searching local artifacts…";
  $("search-results").replaceChildren();
  try {
    const parameters = new URLSearchParams({ operation: picker.operation, field: picker.descriptor.key, q: $("artifact-query").value, limit: "50" });
    for (const folder of picker.folders) parameters.append("folder", folder.id);
    const data = await api(`/artifacts/search?${parameters}`);
    if (sequence !== state.searchSequence || state.picker !== picker) return;
    if (!data.folders && picker.folders.length) throw new Error("Folder filtering is unavailable on this Engine. Clear filters and try again.");
    const results = $("search-results");
    results.replaceChildren();
    picker.folderSuggestions = data.folders ?? [];
    renderFolderFilters();
    const indexed = `${data.index.files} files · ${data.index.directories} folders indexed`;
    const empty = picker.folders.length ? "No results in these folders. Remove a folder filter or change your search." : "No results. Try a shorter filename or add a model folder.";
    $("search-summary").textContent = data.total ? `${data.results.length} of ${data.total} results · ${indexed}` : `${empty} · ${indexed}`;
    if (data.index.issues.length) $("search-summary").append(document.createTextNode(` · ${data.index.issues.map((issue) => issue.message).join("; ")}`));
    for (const candidate of data.results) {
      const info = element("div", {}, [
        element("strong", { text: candidate.relative_path === "." ? candidate.absolute_path : candidate.relative_path }),
        element("p", { text: `${candidate.root_name} · ${candidate.kind}` }),
        element("p", { text: candidate.absolute_path }),
        pill(candidate.structural_status === "resolved" ? "Structure found" : "Structure incomplete", candidate.structural_status === "resolved" ? "" : "warning"),
      ]);
      const missing = candidate.checks.filter((check) => !check.present);
      if (missing.length) info.append(element("div", { class: "missing-checks", text: `Missing: ${missing.map((check) => check.path).join(", ")}` }));
      results.append(element("div", { class: "search-result" }, [info, element("button", {
        class: "secondary", text: "Select", "aria-label": `Select ${candidate.relative_path}`,
        onclick: () => { picker.select(candidate.artifact); $("picker-dialog").close(); },
      })]));
    }
  } catch (error) {
    if (sequence !== state.searchSequence) return;
    $("search-summary").textContent = error.message;
    showError(error);
  }
}

function libraryFamily(document) {
  const family = document.operation.split(".")[0];
  return family === "metaview" ? "qwen2511" : family;
}

const familyNames = {
  ltx23: "LTX 2.3", ltx25: "LTX 2.5", flux2_klein9b: "Klein 9B",
  wan2214b: "Wan 2.2", krea2: "Krea 2 Turbo", qwen2511: "Qwen",
  zimage: "Z-Image Turbo", ideogram4: "Ideogram v4", sdxl: "SDXL", h3: "MiniMax H3",
};

const libraryDivider = $("library-divider");
let libraryWidth = 300;
try { libraryWidth = Number(localStorage.getItem("latentslate.library.width")) || 300; } catch {}
function sizeLibrary(width, remember = false) {
  const maximum = Math.min(440, Math.max(260, window.innerWidth * .42));
  libraryWidth = Math.round(Math.max(260, Math.min(maximum, width)));
  document.querySelector(".workspace").style.setProperty("--library-width", `${libraryWidth}px`);
  libraryDivider.setAttribute("aria-valuenow", libraryWidth);
  libraryDivider.setAttribute("aria-valuemax", Math.round(maximum));
  if (remember) {
    try { localStorage.setItem("latentslate.library.width", String(libraryWidth)); } catch {}
  }
}
libraryDivider.addEventListener("pointerdown", (event) => {
  if (event.button !== 0) return;
  libraryDivider.setPointerCapture(event.pointerId);
  libraryDivider.focus();
  event.preventDefault();
});
libraryDivider.addEventListener("pointermove", (event) => {
  if (libraryDivider.hasPointerCapture(event.pointerId)) sizeLibrary(event.clientX);
});
libraryDivider.addEventListener("pointerup", (event) => {
  if (libraryDivider.hasPointerCapture(event.pointerId)) {
    libraryDivider.releasePointerCapture(event.pointerId);
    sizeLibrary(libraryWidth, true);
  }
});
libraryDivider.addEventListener("keydown", (event) => {
  const widths = { ArrowLeft: libraryWidth - 16, ArrowRight: libraryWidth + 16, Home: 260, End: 440 };
  if (!(event.key in widths)) return;
  event.preventDefault();
  sizeLibrary(widths[event.key], true);
});
libraryDivider.addEventListener("dblclick", () => sizeLibrary(300, true));
window.addEventListener("resize", () => sizeLibrary(libraryWidth));
sizeLibrary(libraryWidth);

function updateNewRecipeModes() {
  const family = $("new-recipe-family").value;
  const choices = state.builtins.filter((item) => libraryFamily(item.document) === family);
  $("new-recipe-mode").replaceChildren(...choices.map((item) => element("option", {
    value: item.key, text: item.document.name.replace(`${familyNames[family] || family} `, ""),
  })));
  $("create-recipe-button").disabled = !choices.length;
}

function openNewRecipe() {
  if (state.busy) return;
  $("new-recipe-form").reset();
  $("new-recipe-error").textContent = "";
  const families = [...new Set(state.builtins.map((item) => libraryFamily(item.document)))];
  $("new-recipe-family").replaceChildren(...families.map((family) => element("option", {
    value: family, text: familyNames[family] || family,
  })));
  updateNewRecipeModes();
  $("new-recipe-dialog").showModal();
  $("new-recipe-name").focus();
}

async function createRecipe(event) {
  event.preventDefault();
  if (state.busy) return;
  const name = $("new-recipe-name").value.trim();
  if (!name) {
    $("new-recipe-error").textContent = "Enter a recipe name.";
    $("new-recipe-name").focus();
    return;
  }
  if (state.dirty && !confirm("Discard unsaved edits and create a new recipe?")) return;
  await work(async () => {
    const controls = [...$("new-recipe-form").elements];
    controls.forEach((control) => { control.disabled = true; });
    $("new-recipe-error").textContent = "";
    try {
      const record = await api(`/builtins/${$("new-recipe-mode").value}/duplicate`, { method: "POST", body: { name } });
      $("new-recipe-dialog").close();
      state.libraryCollapsed.users = false;
      selectRecipe(record);
      await loadLibrary();
      await validate(false);
      notice("Recipe created. Review its model files and parameters, then enable it when ready.");
    } catch (error) {
      if ($("new-recipe-dialog").open) $("new-recipe-error").textContent = error.message;
      else throw error;
    } finally {
      controls.forEach((control) => { control.disabled = false; });
    }
  });
}

$("new-recipe-button").addEventListener("click", openNewRecipe);
$("new-recipe-family").addEventListener("change", updateNewRecipeModes);
$("new-recipe-form").addEventListener("submit", createRecipe);
$("new-recipe-dialog").addEventListener("cancel", (event) => { if (state.busy) event.preventDefault(); });
$("recipe-name").addEventListener("input", () => { state.document.name = $("recipe-name").value; markDirty(); });
$("validate-button").addEventListener("click", () => work(() => validate(true)));
$("save-button").addEventListener("click", save);
$("export-button").addEventListener("click", exportRecipe);
$("delete-button").addEventListener("click", deleteRecipe);
$("import-button").addEventListener("click", () => $("import-dialog").showModal());
$("import-files").addEventListener("change", previewImportFiles);
$("duplicate-button").addEventListener("click", () => work(async () => {
  const record = await api(`/builtins/${state.builtinKey}/duplicate`, { method: "POST", body: {} });
  selectRecipe(record);
  await loadLibrary();
  await validate(false);
  notice("User recipe created. Edit its name, artifacts, and policy, then save a new revision.");
}));
$("connection-button").addEventListener("click", openConnection);
$("connection-form").addEventListener("submit", connect);
$("clear-token-button").addEventListener("click", () => { $("engine-token").value = ""; connect(); });
$("roots-button").addEventListener("click", openRoots);
$("picker-roots-button").addEventListener("click", openRoots);
$("root-form").addEventListener("submit", addRoot);
$("refresh-roots-button").addEventListener("click", refreshIndex);
$("refresh-picker-button").addEventListener("click", refreshIndex);
$("artifact-query").addEventListener("input", () => {
  clearTimeout(state.searchTimer); state.searchSequence++;
  state.searchTimer = setTimeout(searchArtifacts, 220);
});
$("picker-dialog").addEventListener("close", () => { state.picker = null; state.searchSequence++; clearTimeout(state.searchTimer); });
$("roots-dialog").addEventListener("close", () => { if (state.picker) searchArtifacts(); });
document.querySelectorAll("[data-close]").forEach((button) => button.addEventListener("click", () => $(button.dataset.close).close()));
window.addEventListener("beforeunload", (event) => { if (state.dirty) { event.preventDefault(); event.returnValue = ""; } });

$("engine-token").value = state.token;
connect();
