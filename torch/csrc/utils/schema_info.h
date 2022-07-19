#pragma once

#include <torch/csrc/jit/frontend/function_schema_parser.h>
#include <unordered_set>

namespace torch {
namespace utils {

/**
 * class SchemaInfo
 *
 * FunctionSchema wrapper that publicizes argument value specific operator
 * behavior (mutation, aliasing, special cases, etc...)
 */

struct TORCH_API SchemaInfo {
 public:
  explicit SchemaInfo(const c10::FunctionSchema& schema)
      : schema_(std::move(schema)), alias_maps_current_(false) {
    initSchemaInfo();
  }
  explicit SchemaInfo(const char* signature)
      : schema_(torch::jit::parseSchema(signature)),
        alias_maps_current_(false) {
    initSchemaInfo();
  }

  bool is_mutable();

  bool is_mutable(size_t index);

  bool is_mutable(c10::string_view name);

  bool is_nondeterministic() const;

  // Returns whether lhs and rhs may alias directly.
  // This does not account for cases where lhs or rhs are a container that
  // may contain elements that alias the other argument.
  // Besides the checks already included in FunctionSchema::may_alias, this
  // method also accounts special aliasing cases causes by aliasing argument
  // values supplied from addArgumentValue.
  bool may_alias(
      const c10::SchemaArgument& lhs,
      const c10::SchemaArgument& rhs);

  // Returns whether lhs and rhs may alias directly or whether lhs/rhs are a
  // container that may contain elements that alias the other argument. Besides
  // the checks already included in FunctionSchema::may_contain_alias, this
  // method also accounts for special aliasing cases causes by aliasing argument
  // values supplied from addArgumentValue. bidirectional = false only returns
  // whether lhs may contain an alias of rhs while bidirectional = true returns
  // both directions.
  bool may_contain_alias(
      const c10::SchemaArgument& lhs,
      const c10::SchemaArgument& rhs,
      bool bidirectional = true);

  void addArgumentValue(const std::string& name, const at::IValue& value);

  void addArgumentValues(
      const std::vector<c10::optional<at::IValue>>& value_list);

  void addArgumentValues(
      const std::unordered_map<std::string, at::IValue>& values);

 private:
  // This function enforces more conservative results when the TORCH_WARN is
  // triggered from above due to duplicates in an argument list
  void ensureConservativity(
      const std::unordered_set<at::Symbol>& duplicates,
      const std::vector<c10::Argument>& arguments_list,
      c10::SchemaArgType type);

  void initSchemaInfo();

  void generateAliasMaps();

  bool mayContainAliasImpl(
      const c10::SchemaArgument& lhs,
      const c10::SchemaArgument& rhs);

  static std::vector<c10::FunctionSchema> getNonDeterministicOps();

  static std::vector<c10::FunctionSchema> getTrainingOps();

  // Set of all wildcard arguments
  std::unordered_set<c10::SchemaArgument> wildcard_set_;

  // Set of all container arguments
  std::unordered_set<c10::SchemaArgument> container_set_;

  // Map of argument IValues
  std::unordered_map<std::string, at::IValue> value_map_;

  // Alias map of inputs with each other
  std::vector<std::unordered_set<size_t>> input_alias_map_;

  // Alias map of outputs to inputs
  std::vector<std::unordered_set<size_t>> output_alias_map_;

  const c10::FunctionSchema schema_;

  bool alias_maps_current_;
};
} // namespace utils
} // namespace torch
