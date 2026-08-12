/* Minimal browser client. Keep endpoint changes in this block. */
const API = Object.freeze({
  catalog: "/v1/authoring/resources",
  suggestId: "/v1/authoring/resources/suggest-id",
  inspect: "/v1/authoring/resources/inspect",
  preview: "/v1/authoring/resources/preview", // pending backend endpoint
  publish: "/v1/authoring/resources",
  validate: "/v1/authoring/resources/validate",
  fetch: "/v1/authoring/resources/fetch",
  status: "/v1/authoring/status",
  capabilities: "/v1/authoring/capabilities",
});

const TOKEN_KEY = "latentslate.authoring.bearer";
const state = { resources: [], selected: null, inspection: null, pinnedInspection: null, preview: null, previewPayload: null, readOnly: false, sourceCliManaged: false, sourceDirty: false, capabilities: null };
const $ = (selector) => document.querySelector(selector);
const form = $("#resource-form");
const fields = Object.fromEntries([...form.elements].filter((el) => el.name).map((el) => [el.name, el]));
const SOURCE_FIELD_NAMES = ["source", "source_type", "revision", "filename", "file_id", "allow_patterns", "ignore_patterns", "token_env", "requires_auth", "expected_size_bytes", "expected_sha256"];

function setNotice(message, isError = false) {
  const node = $("#notice");
  node.textContent = message;
  node.className = `notice${isError ? " error" : ""}`;
  node.hidden = !message;
}
function setErrors(errors) {
  const node = $("#form-errors");
  const items = Array.isArray(errors) ? errors : [errors];
  node.textContent = items.filter(Boolean).join(" ");
  node.hidden = !node.textContent;
}
function setSourceError(message = "") {
  const node = $("#source-error"); node.textContent = message; node.hidden = !message;
}
function setBusy(button, busy, text) {
  if (!button) return;
  if (busy) { button.dataset.label = button.textContent; button.textContent = text || "Working…"; }
  else button.textContent = button.dataset.label || button.textContent;
  button.disabled = busy;
}
function token() { return sessionStorage.getItem(TOKEN_KEY) || ""; }
function askForToken() {
  const value = window.prompt("Authoring API bearer token (stored only for this browser session):", token());
  if (value === null) return false;
  if (value.trim()) sessionStorage.setItem(TOKEN_KEY, value.trim());
  else sessionStorage.removeItem(TOKEN_KEY);
  return true;
}
async function api(path, options = {}, retry = true) {
  const headers = new Headers(options.headers || {});
  headers.set("Accept", "application/json");
  if (options.body) headers.set("Content-Type", "application/json");
  if (token()) headers.set("Authorization", `Bearer ${token()}`);
  let response;
  try { response = await fetch(path, { ...options, headers }); }
  catch (error) { throw new Error(`Network error: ${error.message}`); }
  if (response.status === 401 && retry && askForToken()) return api(path, options, false);
  const body = await response.json().catch(() => null);
  if (!response.ok) throw new Error(body?.detail || body?.message || `${response.status} ${response.statusText}`);
  return body;
}
function scalar(value) { return value === undefined || value === null ? "" : String(value); }
function lineList(value) { return Array.isArray(value) ? value.join("\n") : ""; }
function csv(value) { return Array.isArray(value) ? value.join(", ") : ""; }
function inputValue(name) { return fields[name]?.type === "checkbox" ? fields[name].checked : fields[name]?.value.trim(); }
function setValue(name, value) { if (!fields[name]) return; if (fields[name].type === "checkbox") fields[name].checked = Boolean(value); else fields[name].value = scalar(value); }
function optional(value) { return value === "" || value === undefined ? undefined : value; }
function lines(value) { return value.split(/\r?\n/).map((item) => item.trim()).filter(Boolean); }
function localSourceError(source) {
  const value = source.trim();
  const windowsDrive = /^[a-z]:[\\/]/i.test(value);
  const uncPath = /^\\\\|^\/\//.test(value);
  const relativePath = /^(?:\.\.?[\\/])/.test(value);
  const bareArtifact = !/[\\/:]/.test(value) && /\.(?:safetensors|gguf|ckpt|pt|pth|bin)$/i.test(value);
  if (bareArtifact) return "A filename alone is not globally unique. Paste the full Hugging Face repo/file URL or locator, for example hf://Kutches/Kl4b/70sSciFiKlein9b.safetensors.";
  if (!(windowsDrive || uncPath || relativePath)) return "";
  return "Browser inspection supports remote Hugging Face/CivitAI only. For a local or NAS file, use .\\scripts\\engine.ps1 resources inspect \"<full-path>\"; a full path is required.";
}
function setSelectOptions(select, values, placeholder) {
  const current = select.value;
  select.replaceChildren();
  if (placeholder) { const option = new Option(placeholder, ""); option.disabled = true; select.add(option); }
  [...new Set(values)].filter(Boolean).sort().forEach((value) => select.add(new Option(value, value)));
  select.value = values.includes(current) ? current : "";
}
function availableFamilies() {
  const resourceAuthoring = state.capabilities?.resource_authoring || {};
  const advertised = resourceAuthoring.families;
  const toolFamilies = (state.capabilities?.base_tools || []).map((tool) => tool.family);
  return [...new Set([...(Array.isArray(advertised) ? advertised : toolFamilies), "custom"])];
}
function updateFamilyOptions(preferred = "") {
  const family = fields.family;
  const values = availableFamilies();
  setSelectOptions(family, values, "Choose a family");
  if (values.includes(preferred)) family.value = preferred;
  else if (preferred) { family.add(new Option(`${preferred} (catalog value)`, preferred)); family.value = preferred; }
}
function updateTokenEnvironment() {
  const sourceType = inputValue("source_type"); const tokenEnv = fields.token_env;
  const allowed = sourceType === "civitai" ? ["", "CIVITAI_TOKEN"] : sourceType === "huggingface" ? ["", "HF_TOKEN"] : [""];
  setSelectOptions(tokenEnv, allowed.filter(Boolean), "None");
  tokenEnv.disabled = sourceType === "auto";
  if (sourceType === "auto") tokenEnv.value = "";
}
function syncKindConstraints() {
  const isLora = inputValue("kind") === "lora";
  fields.base_model.required = isLora;
  $("#base-model-required").hidden = !isLora;
  if (state.inspection) showRequiredFields();
}
function requiredDeclarationFields() {
  const names = ["kind", "resource_id", "family", "name"];
  if (inputValue("kind") === "lora") names.push("base_model");
  return names;
}
function showRequiredFields() {
  const missing = requiredDeclarationFields().filter((name) => !inputValue(name));
  requiredDeclarationFields().forEach((name) => {
    const field = fields[name]; const invalid = missing.includes(name);
    field.classList.toggle("field-invalid", invalid); field.setAttribute("aria-invalid", String(invalid));
  });
  const node = $("#required-fields");
  if (missing.length) {
    const labels = { kind: "Kind", resource_id: "Resource ID", family: "Family", name: "Name", base_model: "Base model" };
    node.textContent = `Required before preview: ${missing.map((name) => labels[name]).join(", ")}.`;
  }
  node.hidden = !missing.length;
}
function isLocal(record) {
  return record.editable === true || record.declaration_origin === "local";
}
function availability(record) { return record.available === false ? `missing: ${record.unavailable_reason || "unavailable"}` : "installed"; }
function ownershipLabel(record) {
  if (record.editable === true || record.declaration_origin === "local") return "LOCAL · EDITABLE";
  if (record.declaration_origin === "builtin") return "BUILT-IN · VIEW ONLY";
  return "DISCOVERED · VIEW ONLY";
}
function sourceFromDescriptor(record) {
  const source = (record.sources || [])[0] || {};
  const metadata = record.metadata || {};
  const type = source.type || record.metadata?.authoring_source_type || "auto";
  let endpointSource = metadata.authoring_canonical_source;
  if (!endpointSource || endpointSource === "local-import") {
    if (source.type === "huggingface" && source.repo_id) {
      endpointSource = `hf://${source.repo_id}${source.filename ? `/${source.filename}` : ""}${source.revision ? `@${source.revision}` : ""}`;
    } else if (source.type === "civitai" && source.model_version_id) {
      endpointSource = `civitai://version/${source.model_version_id}`;
    } else endpointSource = source.url || "";
  }
  return {
    source: endpointSource === "local-import" ? "" : endpointSource,
    source_type: type === "manual" ? "auto" : type,
    revision: source.revision, filename: source.filename, file_id: source.file_id,
    allow_patterns: lineList(source.allow_patterns), ignore_patterns: lineList(source.ignore_patterns),
    token_env: source.token_env, requires_auth: source.requires_auth,
    expected_sha256: source.sha256, expected_size_bytes: record.size_bytes,
  };
}
function resetEditor() {
  form.reset(); state.selected = null; state.inspection = null; state.pinnedInspection = null; state.preview = null; state.previewPayload = null; state.readOnly = false; state.sourceCliManaged = false; state.sourceDirty = false;
  $("#definition-fieldset").disabled = true; $("#facts-fieldset").hidden = true;
  $("#declaration-preview").hidden = true; $("#action-bar").hidden = true;
  $("#candidate-picker").hidden = true; $("#cli-source-note").hidden = true;
  $("#inspection-state").textContent = ""; setErrors("");
  setSourceError("");
  $("#required-fields").hidden = true;
  $("#delete-actions").hidden = true;
  updateFamilyOptions(); updateTokenEnvironment(); syncKindConstraints();
}
function renderCatalog() {
  const list = $("#resource-list"); list.replaceChildren();
  const grouped = new Map();
  state.resources.forEach((resource) => { const family = resource.family || "unclassified"; (grouped.get(family) || grouped.set(family, []).get(family)).push(resource); });
  [...grouped.keys()].sort().forEach((family) => {
    const section = document.createElement("section"); section.className = "family-group";
    const heading = document.createElement("h3"); heading.textContent = family; section.append(heading);
    grouped.get(family).sort((a, b) => (a.name || a.id).localeCompare(b.name || b.id)).forEach((resource) => {
      const button = document.createElement("button"); button.type = "button"; button.className = "resource-item";
      button.setAttribute("aria-current", state.selected?.id === resource.id ? "true" : "false");
      const availabilityClass = resource.available === false ? "bad" : "good";
      button.innerHTML = `<strong>${escapeHtml(resource.name || resource.id)}</strong><small>${escapeHtml(resource.id)}</small><small class="row-status"><span class="ownership">${ownershipLabel(resource)}</span><span class="availability ${availabilityClass}">${escapeHtml(availability(resource))}</span></small>`;
      button.addEventListener("click", () => selectResource(resource)); section.append(button);
    }); list.append(section);
  });
  $("#catalog-summary").textContent = `${state.resources.length} resource${state.resources.length === 1 ? "" : "s"} · grouped by family`;
}
function escapeHtml(text) { const span = document.createElement("span"); span.textContent = text; return span.innerHTML; }
function showFacts(inspection) {
  const facts = inspection?.facts || {};
  const detected = inspection?.detected || {};
  const pairs = {
    "Canonical source": inspection?.canonical_source, "Source type": inspection?.source_type,
    Filename: facts.filename, "Size bytes": facts.size_bytes, SHA256: facts.sha256,
    Format: facts.format, Precision: facts.precision, Quantization: facts.quantization,
    "Detected": Object.keys(detected).length ? JSON.stringify(detected) : "",
  };
  const list = $("#facts-list"); list.replaceChildren();
  Object.entries(pairs).forEach(([key, value]) => { if (value === undefined || value === null || value === "") return; const dt = document.createElement("dt"); dt.textContent = key; const dd = document.createElement("dd"); dd.textContent = typeof value === "object" ? JSON.stringify(value) : String(value); list.append(dt, dd); });
  $("#facts-fieldset").hidden = !list.children.length;
}
function setReadOnly(readOnly) {
  state.readOnly = readOnly;
  $("#definition-fieldset").disabled = false;
  [...$("#definition-fieldset").querySelectorAll("input, select, textarea")].forEach((node) => { node.disabled = readOnly; });
  $("#source-fieldset").disabled = readOnly;
  $("#action-bar").hidden = false;
  $("#publish-button").disabled = true;
  $("#validate-button").disabled = false;
  $("#fetch-button").disabled = false;
  $("#delete-actions").hidden = readOnly || !state.selected || !isLocal(state.selected);
  $("#edit-help").textContent = readOnly ? "Built-in declarations are read-only. Validation and fetch remain available." : "Editable values are used to generate the resource declaration.";
}
function fillDefinition(record) {
  updateFamilyOptions(record.family);
  setValue("resource_id", record.id || record.resource_id);
  ["kind", "family", "name", "relative_path", "format", "precision", "quantization", "base_model", "component", "description"].forEach((key) => setValue(key, record[key]));
  setValue("tags", csv(record.tags)); setValue("metadata", JSON.stringify(record.metadata || {}, null, 2));
  const source = sourceFromDescriptor(record); Object.entries(source).forEach(([key, value]) => setValue(key, value)); updateTokenEnvironment(); setValue("token_env", source.token_env);
  syncKindConstraints();
}
function isCliManagedSource(record) {
  const source = (record.sources || [])[0] || {};
  return source.type === "manual" || sourceFromDescriptor(record).source === "";
}
function selectResource(record) {
  resetEditor(); state.selected = record; state.sourceCliManaged = isCliManagedSource(record); fillDefinition(record); setReadOnly(!isLocal(record));
  $("#editor-kicker").textContent = isLocal(record) ? "Local declaration" : "Built-in declaration";
  $("#editor-heading").textContent = record.name || record.id; $("#entry-state").textContent = availability(record); $("#entry-state").className = `state${record.available === false ? " bad" : " good"}`;
  $("#cli-source-note").hidden = !state.sourceCliManaged;
  if (state.sourceCliManaged) {
    $("#source-fieldset").disabled = true;
    $("#inspection-state").textContent = "CLI-managed source; no server re-inspection.";
  }
  showFacts(record.inspection || null); renderCatalog(); document.querySelector(".workspace").classList.add("show-editor");
}
function createResource() {
  resetEditor(); $("#editor-kicker").textContent = "New local resource"; $("#editor-heading").textContent = "Inspect a source"; $("#entry-state").textContent = "Draft"; $("#entry-state").className = "state"; document.querySelector(".workspace").classList.add("show-editor"); $("#source").focus(); renderCatalog();
}
function clearPreview() {
  state.preview = null; state.previewPayload = null; $("#publish-button").disabled = true; $("#declaration-preview").hidden = true;
  if (state.inspection) showRequiredFields();
}
function previewIsValid() {
  return Boolean(state.preview) && state.preview.valid !== false && !(state.preview.errors || []).length;
}
function inspectRequest() {
  return Object.fromEntries(Object.entries({
    source: inputValue("source"), source_type: inputValue("source_type"), revision: inputValue("revision"), filename: inputValue("filename"),
    file_id: inputValue("file_id") ? Number(inputValue("file_id")) : undefined, allow_patterns: lines(inputValue("allow_patterns")), ignore_patterns: lines(inputValue("ignore_patterns")),
    token_env: inputValue("token_env"), requires_auth: inputValue("requires_auth"), expected_size_bytes: inputValue("expected_size_bytes") ? Number(inputValue("expected_size_bytes")) : undefined,
    expected_sha256: inputValue("expected_sha256"),
  }).filter(([, value]) => value !== undefined && value !== "" && !(Array.isArray(value) && value.length === 0)));
}
function pinnedRequest(result) {
  const exact = result.exact_source;
  if (!exact) return null;
  const request = {
    source_type: exact.type,
    revision: exact.revision, filename: exact.filename, file_id: exact.file_id,
    allow_patterns: exact.allow_patterns || [], ignore_patterns: exact.ignore_patterns || [],
    token_env: exact.token_env, requires_auth: exact.requires_auth,
    expected_size_bytes: result.facts?.size_bytes, expected_sha256: result.facts?.sha256 || exact.sha256,
  };
  if (exact.type === "huggingface" && exact.repo_id) request.source = `hf://${exact.repo_id}${exact.filename ? `/${exact.filename}` : ""}${exact.revision ? `@${exact.revision}` : ""}`;
  else if (exact.type === "civitai" && exact.model_version_id) request.source = `civitai://version/${exact.model_version_id}`;
  else request.source = exact.url || result.canonical_source;
  return Object.fromEntries(Object.entries(request).filter(([, value]) => value !== undefined && value !== "" && !(Array.isArray(value) && value.length === 0)));
}
function applyPinnedInspection(request) {
  if (!request) return;
  setValue("source", request.source); setValue("source_type", request.source_type); updateTokenEnvironment();
  ["revision", "filename", "file_id", "allow_patterns", "ignore_patterns", "token_env", "requires_auth", "expected_size_bytes", "expected_sha256"].forEach((name) => {
    const value = request[name]; setValue(name, Array.isArray(value) ? lineList(value) : value);
  });
}
function sourceNeedsInspection() {
  return !(state.selected && isLocal(state.selected) && !state.sourceDirty);
}
function markSourceDirty() {
  if (state.readOnly || state.sourceCliManaged) return;
  state.sourceDirty = true; state.inspection = null; state.pinnedInspection = null; clearPreview();
  $("#inspection-state").textContent = "Source changed. Inspect it again before previewing or publishing.";
  $("#facts-fieldset").hidden = true;
}
function declarationRequest() {
  let metadata = {};
  const rawMetadata = inputValue("metadata");
  if (rawMetadata) { try { metadata = JSON.parse(rawMetadata); } catch { throw new Error("Metadata must be a JSON object."); } }
  if (typeof metadata !== "object" || Array.isArray(metadata) || metadata === null) throw new Error("Metadata must be a JSON object.");
  const payload = { resource_id: inputValue("resource_id"), kind: inputValue("kind"), family: inputValue("family"), name: optional(inputValue("name")), relative_path: optional(inputValue("relative_path")), format: optional(inputValue("format")), precision: optional(inputValue("precision")), quantization: optional(inputValue("quantization")), base_model: optional(inputValue("base_model")), component: optional(inputValue("component")), description: optional(inputValue("description")), tags: inputValue("tags").split(",").map((item) => item.trim()).filter(Boolean), metadata };
  if (state.sourceCliManaged) {
    if (!(state.selected && isLocal(state.selected))) throw new Error("A CLI-managed source can only update an existing local declaration.");
  } else if (sourceNeedsInspection()) {
    if (!state.pinnedInspection || !state.inspection?.exact_source) throw new Error("Inspect the current source and select an exact file before previewing or publishing.");
    payload.inspection = state.pinnedInspection;
  }
  return payload;
}
function errorsFrom(error) { return error?.message || String(error); }
async function inspectSource() {
  const button = $("#inspect-button"); setErrors("");
  if (!inputValue("source")) { setSourceError("Enter a remote Hugging Face or CivitAI locator before inspection."); setErrors("Source is required before inspection."); $("#source").focus(); return; }
  const localError = localSourceError(inputValue("source"));
  if (localError) { setSourceError(localError); setErrors(localError); $("#source").focus(); return; }
  setSourceError("");
  setBusy(button, true, "Inspecting…");
  try {
    const result = await api(API.inspect, { method: "POST", body: JSON.stringify(inspectRequest()) });
    state.inspection = result; state.pinnedInspection = pinnedRequest(result); state.sourceDirty = true; clearPreview(); applyPinnedInspection(state.pinnedInspection); $("#definition-fieldset").disabled = false; setReadOnly(false);
    const recommended = result.recommended || {}; setValue("name", recommended.name || inputValue("name"));
    updateFamilyOptions(recommended.family || inputValue("family")); setValue("family", recommended.family || inputValue("family")); setValue("component", recommended.component || inputValue("component"));
    if (recommended.base_model) setValue("base_model", recommended.base_model);
    if (recommended.component === "lora") setValue("kind", "lora");
    syncKindConstraints();
    setValue("format", result.facts?.format === "unknown" ? "" : result.facts?.format); setValue("precision", result.facts?.precision === "unknown" ? "" : result.facts?.precision); setValue("quantization", result.facts?.quantization === "unknown" ? "" : result.facts?.quantization);
    if (inputValue("kind") && !inputValue("resource_id")) await suggestId();
    showFacts(result); $("#inspection-state").textContent = result.warnings?.length ? `Inspected with ${result.warnings.length} warning(s).` : "Inspection complete.";
    $("#edit-help").textContent = "Inspection-prefilled values are inferred and editable. Verified artifact facts are kept below.";
    if (result.warnings?.length) setNotice(result.warnings.join(" "));
    renderCandidates(result.candidates || []);
    $("#editor-heading").textContent = "Define local resource"; $("#entry-state").textContent = "Inspected"; $("#entry-state").className = "state good";
    showRequiredFields();
  } catch (error) { setErrors(errorsFrom(error)); } finally { setBusy(button, false); }
}
function renderCandidates(candidates) {
  const picker = $("#candidate-picker"); const select = $("#candidate-file-id"); select.replaceChildren();
  if (!candidates.length || state.inspection?.exact_source) { picker.hidden = true; return; }
  candidates.forEach((candidate) => {
    const option = document.createElement("option"); option.value = candidate.id;
    const details = [candidate.filename, candidate.size_bytes ? `${candidate.size_bytes} bytes` : ""].filter(Boolean).join(" · ");
    option.textContent = `${candidate.label || candidate.id}${details ? ` — ${details}` : ""}`; select.append(option);
  });
  picker.hidden = false;
}
async function inspectCandidate() {
  const value = $("#candidate-file-id").value;
  if (!/^\d+$/.test(value)) { setErrors("The selected candidate has no numeric file ID for this source type."); return; }
  setValue("file_id", value); await inspectSource();
}
async function suggestId() {
  const request = { kind: inputValue("kind"), family: inputValue("family"), name: inputValue("name"), source: inputValue("source") };
  if (!request.kind || !request.family || !request.name || !request.source) return;
  try {
    const response = await api(API.suggestId, { method: "POST", body: JSON.stringify(request) });
    setValue("resource_id", response.resource_id || response.id || response.suggested_id || "");
  } catch (error) {
    // This endpoint is intentionally forward-compatible. A manual ID remains valid.
    $("#inspection-state").textContent = "Inspection complete. Enter a resource ID to continue.";
  }
}
async function validateResource() {
  const button = $("#validate-button"); setErrors(""); setBusy(button, true, "Validating…");
  try {
    if (state.readOnly) {
      const id = state.selected?.id || inputValue("resource_id");
      const response = await api(`${API.validate}?resource_id=${encodeURIComponent(id)}`);
      $("#publish-button").disabled = true;
      const errors = response.errors || [];
      if (!response.valid || errors.length) setErrors(errors.length ? errors : `Validation failed for ${id}.`);
      else setNotice(`Built-in resource ${id} is valid.`);
      return;
    }
    showRequiredFields();
    if (!form.checkValidity()) { form.reportValidity(); return; }
    const request = declarationRequest();
    const existing = state.selected && isLocal(state.selected) ? state.selected.id : "";
    const previewPath = existing ? `${API.preview}?existing_resource_id=${encodeURIComponent(existing)}` : API.preview;
    const previewPayload = { ...request, replace: Boolean(existing) };
    const response = await api(previewPath, { method: "POST", body: JSON.stringify(previewPayload) });
    const errors = response.errors || []; const valid = response.valid !== false && errors.length === 0;
    state.preview = valid ? response : null; state.previewPayload = valid ? previewPayload : null;
    $("#publish-button").disabled = !valid || state.readOnly;
    if (!valid) setErrors(errors.length ? errors : "Preview failed.");
    else setNotice(`Preview passed for ${response.resource?.id || response.descriptor?.id || inputValue("resource_id")}.`);
    if (response.warnings?.length) setNotice(response.warnings.join(" "));
    renderPreview(response);
  } catch (error) { setErrors(errorsFrom(error)); } finally { setBusy(button, false); }
}
function renderPreview(response) {
  const declaration = response.toml || response.declaration || response.generated_declaration || response.preview;
  if (!declaration) return;
  $("#preview-content").textContent = typeof declaration === "string" ? declaration : JSON.stringify(declaration, null, 2); $("#declaration-preview").hidden = false;
}
function sameJson(left, right) { return JSON.stringify(left) === JSON.stringify(right); }
function publicationMatchesPreview(preview, publication) {
  const previewToml = preview.toml || preview.declaration || preview.generated_declaration;
  const publicationToml = publication.toml || publication.declaration || publication.generated_declaration;
  if (previewToml && publicationToml && previewToml !== publicationToml) return false;
  const previewResource = preview.resource || preview.descriptor;
  if (previewResource && publication.resource && !sameJson(previewResource, publication.resource)) return false;
  if (preview.inspection?.exact_source && publication.inspection?.exact_source && !sameJson(preview.inspection.exact_source, publication.inspection.exact_source)) return false;
  return true;
}
async function publishResource() {
  if (state.readOnly) return;
  const button = $("#publish-button"); setErrors(""); let payload;
  if (!form.checkValidity()) { form.reportValidity(); return; }
  if (!previewIsValid()) { setErrors("Run a successful Validate / preview before publishing."); return; }
  try { payload = declarationRequest(); } catch (error) { setErrors(errorsFrom(error)); return; }
  const existing = state.resources.find((resource) => resource.id === payload.resource_id);
  const replacing = Boolean(existing && isLocal(existing));
  if (replacing && !window.confirm(`Replace the existing local declaration for ${payload.resource_id}? This overwrites its declaration but does not delete artifacts.`)) return;
  payload.replace = replacing;
  if (!sameJson(payload, state.previewPayload)) { setErrors("The form changed after preview. Run Validate / preview again before publishing."); return; }
  setBusy(button, true, "Publishing…");
  try {
    const endpoint = replacing ? `${API.catalog}/${encodeURIComponent(payload.resource_id)}` : API.publish;
    const response = await api(endpoint, { method: replacing ? "PUT" : "POST", body: JSON.stringify(payload) });
    renderPreview(response); const activation = response.activation || {};
    if (!publicationMatchesPreview(state.preview, response)) throw new Error("Published declaration differs from the validated preview. The change may be on disk; review the returned catalog state before retrying.");
    setNotice(`Published ${response.resource?.id || payload.resource_id}. ${activation.required_action === "restart_engine" ? "Restart Engine to activate this catalog change." : "Active."}`);
    if (response.resource) state.selected = response.resource;
    await loadCatalog();
  } catch (error) { setErrors(errorsFrom(error)); } finally { setBusy(button, false); }
}
async function fetchResource() {
  const id = inputValue("resource_id"); if (!id) { setErrors("Select or define a resource ID before fetch."); return; }
  const button = $("#fetch-button"); setErrors(""); setBusy(button, true, "Fetching…");
  try { const result = await api(`${API.fetch}?resource_id=${encodeURIComponent(id)}`, { method: "POST" }); setNotice(`Fetch completed for ${id}. ${result.message || ""}`); await loadCatalog(); }
  catch (error) { setErrors(errorsFrom(error)); } finally { setBusy(button, false); }
}
async function deleteResource(deleteArtifact) {
  const id = state.selected?.id;
  if (!id || state.readOnly || !isLocal(state.selected)) return;
  const message = deleteArtifact
    ? `Permanently remove the local declaration and downloaded artifact for ${id}? This is blocked if any recipe or draft depends on it.`
    : `Remove the local declaration for ${id}? The installed artifact will be kept and may reappear as a read-only discovered resource. This is blocked if any recipe or draft depends on it.`;
  if (!window.confirm(message)) return;
  const button = deleteArtifact ? $("#delete-resource-button") : $("#remove-declaration-button");
  setErrors(""); setBusy(button, true, deleteArtifact ? "Deleting…" : "Removing…");
  try {
    const result = await api(`${API.catalog}/${encodeURIComponent(id)}`, {
      method: "DELETE",
      body: JSON.stringify({ delete_artifact: deleteArtifact }),
    });
    const suffix = result.resulting_resource
      ? " The remaining artifact is still visible as a read-only discovered resource."
      : "";
    const warnings = (result.warnings || []).join(" ");
    setNotice(`Removed ${id}.${suffix}${warnings ? ` ${warnings}` : ""}`);
    createResource();
    await loadCatalog();
  } catch (error) { setErrors(errorsFrom(error)); } finally { setBusy(button, false); }
}
async function loadCatalog() {
  $("#catalog-state").textContent = "Loading…";
  try {
    const [status, capabilities, catalog] = await Promise.all([api(API.status), api(API.capabilities).catch(() => null), api(API.catalog)]);
    state.capabilities = capabilities; updateFamilyOptions(); state.resources = catalog.resources || catalog.items || catalog || [];
    if (!Array.isArray(state.resources)) state.resources = [];
    renderCatalog(); const stale = status?.stale;
    $("#catalog-state").textContent = stale ? "Restart required" : "Catalog loaded"; $("#catalog-state").className = `state ${stale ? "bad" : "good"}`;
    if (stale) setNotice("Catalog changes are on disk. Restart Engine to activate them.");
  } catch (error) { $("#catalog-state").textContent = "Unavailable"; $("#catalog-state").className = "state bad"; $("#catalog-summary").textContent = "Catalog could not be loaded."; setNotice(errorsFrom(error), true); }
}

$("#create-button").addEventListener("click", createResource); $("#back-button").addEventListener("click", () => document.querySelector(".workspace").classList.remove("show-editor"));
$("#inspect-button").addEventListener("click", inspectSource); $("#candidate-inspect-button").addEventListener("click", inspectCandidate); $("#validate-button").addEventListener("click", validateResource); $("#publish-button").addEventListener("click", publishResource); $("#fetch-button").addEventListener("click", fetchResource);
$("#remove-declaration-button").addEventListener("click", () => deleteResource(false)); $("#delete-resource-button").addEventListener("click", () => deleteResource(true));
$("#reload-button").addEventListener("click", loadCatalog); $("#token-button").addEventListener("click", () => { if (askForToken()) loadCatalog(); });
$("#source_type").addEventListener("change", updateTokenEnvironment);
$("#kind").addEventListener("change", async () => { syncKindConstraints(); clearPreview(); if (state.inspection && !inputValue("resource_id")) await suggestId(); });
$("#source").addEventListener("input", () => setSourceError(""));
SOURCE_FIELD_NAMES.forEach((name) => {
  fields[name].addEventListener("input", markSourceDirty);
  fields[name].addEventListener("change", markSourceDirty);
});
form.addEventListener("input", clearPreview); form.addEventListener("change", clearPreview);
createResource(); loadCatalog();
