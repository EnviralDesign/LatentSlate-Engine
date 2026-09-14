function newItem(descriptor) {
  if (descriptor.value_type === "artifact") return { source: "local", path: "" };
  if (descriptor.value_type === "adapter") return { artifact: { source: "local", path: "" }, strength: 1 };
  if (["number", "integer"].includes(descriptor.value_type)) return 1;
  if (descriptor.value_type === "boolean") return false;
  return "";
}

export function appendCollectionItem(members) {
  for (const { descriptor, field } of members) {
    field.value = [...(Array.isArray(field.value) ? field.value : []), newItem(descriptor)];
  }
}

export function moveCollectionItem(members, index, offset) {
  for (const { field } of members) {
    [field.value[index], field.value[index + offset]] = [field.value[index + offset], field.value[index]];
  }
}

export function removeCollectionItem(members, index) {
  for (const { field } of members) field.value.splice(index, 1);
}
