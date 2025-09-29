# -*- coding: utf-8 -*-
# 更新 Java 代码以使用 SLF4J 日志记录

import os
import re


# [之前定义的所有辅助函数，如 find_java_files, get_class_name, add_slf4j_imports,
#  add_logger_field, replace_system_out_println, replace_exception_print_stack_trace
#  都应该在这里。refactor_log4j_to_slf4j 函数将被移除。]

def find_java_files(directory):
    """
    递归查找指定目录下的所有 .java 文件。
    """
    java_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".java"):
                java_files.append(os.path.join(root, file))
    return java_files


def get_class_name(file_path):
    """
    从 Java 文件路径中提取类名。
    """
    base_name = os.path.basename(file_path)
    class_name, _ = os.path.splitext(base_name)
    return class_name


def add_slf4j_imports(content_lines, class_name_for_info):
    """
    如果 SLF4J 的导入语句不存在，则添加它们。
    """
    import_logger = "import org.slf4j.Logger;"
    import_logger_factory = "import org.slf4j.LoggerFactory;"

    has_logger_import = any(import_logger in line for line in content_lines)
    has_logger_factory_import = any(import_logger_factory in line for line in content_lines)

    already_fully_imported = has_logger_import and has_logger_factory_import

    if already_fully_imported:
        return content_lines, False

    new_imports_to_add = []
    if not has_logger_import:
        new_imports_to_add.append(import_logger + "\n")
    if not has_logger_factory_import:
        new_imports_to_add.append(import_logger_factory + "\n")

    if not new_imports_to_add:
        return content_lines, False

    package_line_index = -1
    last_import_line_index = -1
    first_code_line_after_potential_comment_index = 0

    in_block_comment_header = False
    for i, line in enumerate(content_lines):
        stripped_line = line.strip()
        if i == 0 and stripped_line.startswith("/**"):
            in_block_comment_header = True
            continue
        if in_block_comment_header and stripped_line.endswith("*/"):
            first_code_line_after_potential_comment_index = i + 1
            in_block_comment_header = False
            continue
        if not in_block_comment_header and stripped_line:
            first_code_line_after_potential_comment_index = i
            break
        if not stripped_line and i > first_code_line_after_potential_comment_index:
            first_code_line_after_potential_comment_index = i

    for i, line in enumerate(content_lines):
        line_s = line.strip()
        if line_s.startswith("package "):
            package_line_index = i
        if line_s.startswith("import "):
            last_import_line_index = i

    insert_at_index = 0
    added_newline_before_imports = False

    if last_import_line_index != -1:
        insert_at_index = last_import_line_index + 1
    elif package_line_index != -1:
        insert_at_index = package_line_index + 1
        is_next_line_empty_or_import = False
        if insert_at_index < len(content_lines):
            next_line_strip = content_lines[insert_at_index].strip()
            if not next_line_strip or next_line_strip.startswith("import "):
                is_next_line_empty_or_import = True

        if not is_next_line_empty_or_import:
            new_imports_to_add.insert(0, "\n")
            added_newline_before_imports = True
    else:
        insert_at_index = first_code_line_after_potential_comment_index
        if insert_at_index == 0 and len(content_lines) > 0:
            stripped_first_line = content_lines[0].strip()
            if stripped_first_line.startswith("public class") or \
                    stripped_first_line.startswith("class") or \
                    stripped_first_line.startswith("public interface") or \
                    stripped_first_line.startswith("interface") or \
                    stripped_first_line.startswith("public enum") or \
                    stripped_first_line.startswith("enum"):
                pass

    result_lines = list(content_lines)

    for i, imp_line in enumerate(new_imports_to_add):
        result_lines.insert(insert_at_index + i, imp_line)

    idx_after_new_imports = insert_at_index + len(new_imports_to_add)

    idx_last_actual_import = idx_after_new_imports - 1
    if added_newline_before_imports and new_imports_to_add and new_imports_to_add[0] == "\n":
        pass

    if idx_last_actual_import >= 0 and idx_last_actual_import < len(result_lines) and \
            result_lines[idx_last_actual_import].strip() and \
            not result_lines[idx_last_actual_import].endswith("\n"):
        result_lines[idx_last_actual_import] += "\n"

    if idx_after_new_imports < len(result_lines):
        line_after_imports_stripped = result_lines[idx_after_new_imports].strip()
        prev_line_is_empty_new_import = added_newline_before_imports and \
                                        new_imports_to_add and \
                                        new_imports_to_add[0] == "\n" and \
                                        idx_after_new_imports == (insert_at_index + 1)

        if line_after_imports_stripped and \
                not line_after_imports_stripped.startswith("import ") and \
                not prev_line_is_empty_new_import:
            if idx_last_actual_import >= 0 and \
                    idx_last_actual_import < len(result_lines) and \
                    result_lines[idx_last_actual_import].strip() and \
                    not result_lines[idx_last_actual_import].endswith("\n\n"):
                result_lines.insert(idx_after_new_imports, "\n")

    if new_imports_to_add:
        print(f"    在 {class_name_for_info} 中添加了缺失的 SLF4J import 语句。")
        return result_lines, True
    return result_lines, False


def add_logger_field(content_lines, class_name):
    """
    如果 SLF4J 的 Logger 成员变量不存在，则添加到类中。变量名固定为 'logger'。
    """
    logger_field_pattern_check_slf4j = r"(private|protected|public)?\s+static\s+(final\s+)?(org\.slf4j\.)?Logger\s+logger\s*=\s*org\.slf4j\.LoggerFactory\.getLogger\("
    generic_logger_check = r"(private|protected|public)?\s+static\s+(final\s+)?Logger\s+logger\s*="

    if any(re.search(logger_field_pattern_check_slf4j, line) for line in content_lines) or \
            any(re.search(generic_logger_check, line) and "LoggerFactory.getLogger(" in line for line in content_lines):
        return content_lines, False

    class_declaration_index = -1
    class_curly_brace_index = -1

    entity_pattern_regex = r"^\s*(public|protected|private)?\s*(static|final|abstract|sealed|non-sealed)?\s*\b(class|interface|enum)\b\s+" + re.escape(
        class_name) + r"\b.*"

    for i, line in enumerate(content_lines):
        if re.search(entity_pattern_regex, line.strip(), re.IGNORECASE):
            class_declaration_index = i
            temp_curly_brace_index = -1
            for j in range(i, len(content_lines)):
                if "{" in content_lines[j]:
                    block_to_check = "".join(content_lines[i: j + 1]).replace("\n", " ")
                    if re.search(entity_pattern_regex, block_to_check.strip(), re.IGNORECASE):
                        temp_curly_brace_index = j
                        break
            if temp_curly_brace_index != -1:
                class_curly_brace_index = temp_curly_brace_index
                break

    if class_curly_brace_index != -1:
        insert_at = class_curly_brace_index + 1

        indentation = "    "
        next_content_line_index = -1
        for k in range(insert_at, len(content_lines)):
            if content_lines[k].strip():
                next_content_line_index = k
                break

        if next_content_line_index != -1:
            match = re.match(r"^(\s+)", content_lines[next_content_line_index])
            if match:
                indentation = match.group(1)
        elif class_curly_brace_index >= 0:
            class_decl_indent_match = re.match(r"^(\s*)", content_lines[class_curly_brace_index])
            if class_decl_indent_match:
                indentation = class_decl_indent_match.group(1) + "    "

        logger_field_declaration = f"{indentation}private static final org.slf4j.Logger logger = org.slf4j.LoggerFactory.getLogger({class_name}.class);\n"

        should_add_blank_line_after = False
        if insert_at < len(content_lines) and \
                content_lines[insert_at].strip() and \
                not content_lines[insert_at].strip().startswith("}"):
            should_add_blank_line_after = True

        result_lines = list(content_lines)
        result_lines.insert(insert_at, logger_field_declaration)

        if should_add_blank_line_after:
            if not logger_field_declaration.endswith("\n\n"):
                blank_line_to_add = "\n"
                result_lines.insert(insert_at + 1, blank_line_to_add)

        print(f"    已向类 {class_name} 添加 SLF4J Logger 成员变量 'logger'。")
        return result_lines, True
    else:
        return content_lines, False


def replace_system_out_println(content_lines, class_name):
    """
    将 System.out.println 替换为 logger.info。
    """
    modified_lines_overall = []
    replacements_total = 0
    pattern_sysout_call = r'System\.out\.println(?=\s*\()'
    file_was_modified_by_replacement = False

    for current_line in content_lines:
        modified_line_content = current_line
        stripped_line_for_comment_check = current_line.lstrip()
        if stripped_line_for_comment_check.startswith("//"):
            modified_lines_overall.append(current_line)
            continue
        if stripped_line_for_comment_check.startswith("/*") or stripped_line_for_comment_check.startswith("* "):
            if not "*/" in stripped_line_for_comment_check:
                modified_lines_overall.append(current_line)
                continue

        new_line, num_line_repl = re.subn(pattern_sysout_call, 'logger.info', current_line)

        if num_line_repl > 0:
            modified_line_content = new_line
            replacements_total += num_line_repl
            file_was_modified_by_replacement = True

        modified_lines_overall.append(modified_line_content)

    if file_was_modified_by_replacement:
        print(f"    在 {class_name} 中，已将 {replacements_total} 处 System.out.println 替换为 logger.info。")
    return modified_lines_overall, replacements_total


def replace_exception_print_stack_trace(content_lines, class_name):
    """
    将 e.printStackTrace(); 替换为 logger.error("{}", e.getMessage(), e);
    """
    modified_lines_overall = []
    replacements_total = 0
    pattern_print_stack_trace = re.compile(r"(\s*)([a-zA-Z0-9_]+)\.printStackTrace\(\s*\)\s*;")
    file_was_modified_by_replacement = False

    for current_line in content_lines:
        modified_line_content = current_line
        stripped_line_for_comment_check = current_line.lstrip()
        if stripped_line_for_comment_check.startswith("//"):
            modified_lines_overall.append(current_line)
            continue
        if stripped_line_for_comment_check.startswith("/*") or stripped_line_for_comment_check.startswith("* "):
            if not "*/" in stripped_line_for_comment_check:
                modified_lines_overall.append(current_line)
                continue

        def replacement_function(match):
            nonlocal file_was_modified_by_replacement
            file_was_modified_by_replacement = True
            indentation = match.group(1)
            exception_variable_name = match.group(2)
            return f'{indentation}logger.error("{{}}", {exception_variable_name}.getMessage(), {exception_variable_name});'

        new_line, num_line_repl = pattern_print_stack_trace.subn(replacement_function, current_line)

        if num_line_repl > 0:
            modified_line_content = new_line
            replacements_total += num_line_repl

        modified_lines_overall.append(modified_line_content)

    if file_was_modified_by_replacement:
        print(f"    在 {class_name} 中，已将 {replacements_total} 处 *.printStackTrace() 替换为 logger.error() 调用。")
    return modified_lines_overall, replacements_total


# (其他函数 find_java_files, get_class_name, add_slf4j_imports,
#  add_logger_field, replace_system_out_println, replace_exception_print_stack_trace 保持不变)

def process_java_file(file_path, encoding='utf-8', dry_run=False):
    """
    处理单个 Java 文件：
    - 如果使用 Log4j，则跳过 SLF4J 相关重构。
    - 如果已完整导入 SLF4J (Logger & LoggerFactory):
        - 若同时存在名为 'logger' 的标准SLF4J字段，则替换 System.out/printStackTrace。
        - 否则 (例如只有imports，或有自定义名称的SLF4J logger)，不添加新logger字段，也不进行替换。
    - 否则 (未使用 Log4j 且 SLF4J imports 不完整)，检查 S.o.p/p.S.t:
        - 若存在，则引入 SLF4J (imports 和 'logger' 字段) 并替换。
        - 若不存在，则不操作。
    """
    print(f"正在处理: {file_path} (使用字符集: {encoding})...")
    try:
        with open(file_path, 'r', encoding=encoding, errors='replace') as f:
            original_lines = f.readlines()
    except Exception as e:
        print(f"    读取文件 {file_path} 错误 (字符集: {encoding}): {e}")
        return

    current_lines = list(original_lines)
    made_any_change_in_file = False
    class_name = get_class_name(file_path)  # 先获取类名，后续可能用到

    # --- 步骤 1: 检测 Log4j 使用 ---
    is_log4j_file = False
    # (Log4j 检测逻辑与上一版本相同，此处省略以保持简洁，实际脚本中应包含)
    log4j_import_pattern = re.compile(r"import\s+org\.apache\.log4j\.(Logger|LogManager|Category)\s*;")
    log4j_explicit_usage_pattern = re.compile(r"org\.apache\.log4j\.(Logger|Category)")
    log4j_getlogger_pattern = re.compile(r"\b(Logger|LogManager|Category)\.getLogger\s*\(")
    file_content_str_for_log4j_check = "".join(original_lines)

    if log4j_import_pattern.search(file_content_str_for_log4j_check) or \
            log4j_explicit_usage_pattern.search(file_content_str_for_log4j_check) or \
            (log4j_getlogger_pattern.search(
                file_content_str_for_log4j_check) and "org.apache.log4j" in file_content_str_for_log4j_check):
        is_log4j_file = True

    if is_log4j_file:
        print(f"  文件 {file_path} 检测到使用 Log4j。将保留其 Log4j 实现，跳过 SLF4J 相关重构。")
        return

        # --- 步骤 2: (非 Log4j 文件) 处理 SLF4J 相关逻辑 ---

    # 检查当前 SLF4J imports 的完整性
    has_slf4j_logger_import = any("import org.slf4j.Logger;" in line for line in current_lines)
    has_slf4j_factory_import = any("import org.slf4j.LoggerFactory;" in line for line in current_lines)
    slf4j_imports_are_complete = has_slf4j_logger_import and has_slf4j_factory_import

    # 检查标准的 'logger' 字段是否存在
    standard_logger_field_exists = False
    # 修正：确保 logger_field_pattern 定义在此处或全局
    logger_field_pattern = r"(?:private|public|protected)?\s*static\s*(?:final\s+)?(?:org\.slf4j\.)?Logger\s+logger\s*=\s*(?:org\.slf4j\.)?LoggerFactory\.getLogger\("
    if any(re.search(logger_field_pattern, line) for line in current_lines):
        standard_logger_field_exists = True

    can_perform_slf4j_replacements = False  # 是否可以安全地进行 S.o.p/p.S.t 替换

    if slf4j_imports_are_complete:
        print(f"  文件 {file_path} 已检测到完整的 SLF4J核心 import 语句。")
        if standard_logger_field_exists:
            print(f"    标准的 'logger' 字段已存在。将尝试替换 System.out/printStackTrace。")
            can_perform_slf4j_replacements = True
            # 不需要调用 add_logger_field 来添加字段
        else:
            # SLF4J imports 完整, 但没有标准的 'logger' 字段。
            # 根据用户要求 "不需要再...创建 logger 对象了" (如果已使用SLF4J)，我们不主动添加 'logger' 字段。
            # 由于没有 'logger' 字段，替换 S.o.p/p.S.t 将没有目标。
            print(
                f"    SLF4J imports 存在，但标准的 'logger' 字段未找到。将不添加新 'logger' 字段，也不进行 S.o.p/p.S.t 替换。")
            can_perform_slf4j_replacements = False
            # 注意: 此处不调用 add_logger_field
    else:
        # SLF4J imports 不完整 (或完全缺失)。
        # 检查是否有 S.o.p 或 p.S.t 需要被替换，如果有，则尝试引入完整的 SLF4J 设置。
        needs_to_introduce_slf4j = False
        print_stack_trace_regex = re.compile(r"[a-zA-Z0-9_]+\.printStackTrace\(\s*\)\s*;")
        for line_content in current_lines:  # 检查当前行（可能是副本）
            stripped_line = line_content.lstrip()
            if stripped_line.startswith("//") or stripped_line.startswith("/*") or stripped_line.startswith("*"):
                continue
            if "System.out.println" in line_content or print_stack_trace_regex.search(line_content):
                needs_to_introduce_slf4j = True
                break

        if needs_to_introduce_slf4j:
            print(
                f"  文件 {file_path} 未完整导入 SLF4J，但检测到 System.out.println 或 printStackTrace。将尝试引入 SLF4J。")

            current_lines_before_imports = list(current_lines)
            current_lines, imports_actually_added = add_slf4j_imports(current_lines, class_name)
            if imports_actually_added or current_lines_before_imports != current_lines:
                made_any_change_in_file = True

            current_lines_before_field = list(current_lines)
            current_lines, logger_field_actually_added = add_logger_field(current_lines, class_name)
            if logger_field_actually_added or current_lines_before_field != current_lines:
                made_any_change_in_file = True

            # 在尝试添加后，再次检查标准 'logger' 字段是否存在，以决定是否可以替换
            if any(re.search(logger_field_pattern, line) for line in current_lines):
                can_perform_slf4j_replacements = True
            else:
                print(f"    虽然尝试引入 SLF4J，但未能成功添加标准的 'logger' 字段。将跳过 S.o.p/p.S.t 替换。")
                can_perform_slf4j_replacements = False
        else:
            print(f"  文件 {file_path} (非Log4j，无完整SLF4J imports) 未包含 S.o.p/p.S.t。无需 SLF4J 操作。")
            # 如果 made_any_change_in_file 仍为 False，则可以安全返回或让后续逻辑处理
            if not made_any_change_in_file: return

    # --- 步骤 3: 如果条件满足，执行 S.o.p 和 p.S.t 的替换 ---
    if can_perform_slf4j_replacements:
        # 替换前再次确保 imports 是完整的 (add_slf4j_imports 是幂等的)
        # 这一步是为了覆盖一种边缘情况：例如，imports_are_complete 为 True，
        # standard_logger_field_exists 也为 True，但可能其中一个 import 语句的格式略有问题
        # 或者在之前的步骤中被意外修改。调用 add_slf4j_imports 可以帮助规范化。
        current_lines_before_final_import_check = list(current_lines)
        current_lines, _ = add_slf4j_imports(current_lines, class_name)
        if current_lines_before_final_import_check != current_lines:
            made_any_change_in_file = True

        current_lines_before_sout = list(current_lines)
        current_lines, replacements_println_made = replace_system_out_println(current_lines, class_name)
        if replacements_println_made > 0 or current_lines_before_sout != current_lines:
            made_any_change_in_file = True

        current_lines_before_pst = list(current_lines)
        current_lines, replacements_stacktrace_made = replace_exception_print_stack_trace(current_lines, class_name)
        if replacements_stacktrace_made > 0 or current_lines_before_pst != current_lines:
            made_any_change_in_file = True

    # --- 步骤 4: 写入文件 (如果发生更改) ---
    if made_any_change_in_file:
        if dry_run:
            print(f"  [演习模式] 文件 {file_path} 将会被修改。")
            # 可选: 打印 diff
            # import difflib
            # diff = difflib.unified_diff("".join(original_lines).splitlines(keepends=True),
            #                             "".join(current_lines).splitlines(keepends=True),
            #                             fromfile='原始文件', tofile='修改后文件', lineterm='')
            # if list(diff): # 仅当有差异时打印
            #    print("拟议的更改:\n" + "".join(list(diff)))
            # else:
            #    print("    内容比较后无实际文本差异。")

        else:
            try:
                if "".join(original_lines) != "".join(current_lines):
                    with open(file_path, 'w', encoding=encoding, errors='replace') as f:
                        f.writelines(current_lines)
                    print(f"  成功处理并保存文件: {file_path} (使用字符集: {encoding})")
                else:
                    # 这个分支很重要，因为 made_any_change_in_file 可能因为函数返回True但实际列表未变而为True
                    print(f"  文件 {file_path} 内容经处理后未发生实际文本变化，未重新保存。")
            except Exception as e:
                print(f"    写入文件 {file_path} 错误 (字符集: {encoding}): {e}")
    else:
        print(f"  文件 {file_path} 未做任何更改。")


# --- main 函数和其余辅助函数保持不变 ---
# (确保将此更新后的 process_java_file 函数放入你的完整脚本中，
#  并确保所有其他辅助函数如 add_slf4j_imports, add_logger_field 等已定义)


def main():
    project_directory = input("请输入你的 Java 项目根目录路径: ")
    if not os.path.isdir(project_directory):
        print(f"错误: 目录 '{project_directory}' 未找到。")
        return

    file_encoding = input("请输入项目 Java 文件的字符集 (例如 UTF-8, GBK) [默认 UTF-8]: ").strip()
    if not file_encoding:
        file_encoding = 'utf-8'
    print(f"将使用字符集: {file_encoding}")

    java_files_to_process = find_java_files(project_directory)

    if not java_files_to_process:
        print(f"在目录 '{project_directory}' 中未找到 .java 文件。")
        return

    print(f"\n找到了 {len(java_files_to_process)} 个 Java 文件待处理。")

    dry_run_choice = input(
        "是否以“演习模式”(DRY RUN)运行？该模式下文件不会被实际修改。(yes/no) [默认 yes]: ").strip().lower()
    is_dry_run = dry_run_choice != "no"

    if is_dry_run:
        print("\n--- 演习模式 (DRY RUN): 文件不会被实际修改。 ---")
    else:
        print("\n--- !!! 实际操作模式 !!! 文件将会被修改。 ---")
        print("--- !!! 请确保你已备份项目或已提交到版本控制系统。 ---")
        confirm = input("是否确定要继续? (yes/no): ").strip().lower()
        if confirm != "yes":
            print("操作已由用户取消。")
            return

    print("\n开始重构过程...\n")
    for java_file_path_item in java_files_to_process:
        process_java_file(java_file_path_item, encoding=file_encoding, dry_run=is_dry_run)
        print("-" * 40)

    print("\n重构过程已完成。")
    if not is_dry_run:
        print("重要提示:")
        print("1. 请务必审查脚本所做的更改，确保一切符合预期。")
        print(
            "2. 对于转换为 SLF4J 的文件，请确保你的项目构建依赖中包含 SLF4J API (slf4j-api.jar) 和一个具体的 SLF4J 实现 (如 logback-classic.jar 或 log4j-slf4j-impl.jar + log4j-core.jar)。")
        print("3. 对于保留 Log4j 的文件，其原有的 Log4j 依赖和配置应保持不变。")


if __name__ == "__main__":
    main()
