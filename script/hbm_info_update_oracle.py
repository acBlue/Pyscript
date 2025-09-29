# -*- coding: utf-8 -*-
#  根据 hbm 文件更新 oracle 的 sequence 信息

import argparse
import os
import xml.etree.ElementTree as ET

import cx_Oracle

# --- Configuration ---
ORACLE_CLIENT_LIB_DIR = r"D:\Program Files\instantclient_21_11"  # Example: r"C:\oracle\instantclient_19_21"

# --- Global Variables ---
table_info_map = {}
sequence_to_tables_map = {}


def parse_hibernate_xml(file_path):
    """
    Parses a single Hibernate XML (.hbm.xml) file to extract table name,
    PK column name, and sequence name.
    """
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        for class_element in root.findall('.//class'):
            table_name = class_element.get('table')
            if not table_name:
                continue

            id_element = class_element.find('./id')
            if id_element is None:
                continue

            pk_column_name = id_element.get('column')
            if not pk_column_name:
                pk_column_name = id_element.get('name')  # Fallback
                if not pk_column_name:
                    print(
                        f"Warning: Primary key column name not found in <id> for table '{table_name}' in '{os.path.basename(file_path)}'. Skipping.")
                    continue

            generator_element = id_element.find('./generator')
            sequence_name = None

            if generator_element is not None and generator_element.get('class') in ['sequence', 'seqhilo']:
                for param_element in generator_element.findall('./param'):
                    if param_element.get('name') == 'sequence' or param_element.get('name') == 'sequence_name':
                        sequence_name = param_element.text
                        break
            elif id_element.find('./sequence') is not None and id_element.find('./sequence').get('name'):
                sequence_element_direct = id_element.find('./sequence')
                sequence_name = sequence_element_direct.get('name')
            else:
                # If no generator or sequence element found, this ID might not be sequence-generated
                # or uses a different mechanism.
                continue  # Skip if no sequence info found for this ID

            if table_name and sequence_name and pk_column_name:
                # Ensure keys are consistently cased for dictionary lookups
                table_name_upper = table_name.upper()
                pk_column_name_upper = pk_column_name.upper()
                sequence_name_upper = sequence_name.upper()

                print(
                    f"Found in XML '{os.path.basename(file_path)}': Table='{table_name_upper}', PK_Column='{pk_column_name_upper}', Sequence='{sequence_name_upper}'")
                table_info_map[table_name_upper] = {
                    'sequence': sequence_name_upper,
                    'pk_column': pk_column_name_upper
                }
            # else:
            # Optional: Add warning if some parts are missing after finding a generator
            # if not sequence_name and generator_element is not None:
            # print(f"Warning: Sequence name parameter not found within <generator> for table '{table_name}', PK column '{pk_column_name}' in XML '{os.path.basename(file_path)}'.")
    except ET.ParseError as e:
        print(f"Error parsing XML file {file_path}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while parsing XML {file_path}: {e}")


def find_and_process_hibernate_configs(root_dir):
    """
    Walks through the directory, parses .hbm.xml files,
    and builds the sequence_to_tables_map.
    """
    global sequence_to_tables_map
    found_files = False
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".hbm.xml"):
                found_files = True
                file_path = os.path.join(subdir, file)
                print(f"Parsing HBM XML: {file_path}")
                parse_hibernate_xml(file_path)

    if not found_files:
        print(f"No '.hbm.xml' files found in the specified directory: {root_dir}")
        return

    if not table_info_map:
        print("No table information extracted from HBM files.")
        return

    for table_name_upper, info in table_info_map.items():
        seq_name_upper = info['sequence']
        pk_col_upper = info['pk_column']
        if seq_name_upper not in sequence_to_tables_map:
            sequence_to_tables_map[seq_name_upper] = []
        sequence_to_tables_map[seq_name_upper].append({'table_name': table_name_upper, 'pk_column': pk_col_upper})

    if sequence_to_tables_map:
        print("\n--- Summary of Sequences and Associated Tables ---")
        for seq, tables in sequence_to_tables_map.items():
            table_details = [f"{t['table_name']} (PK: {t['pk_column']})" for t in tables]
            print(f"Sequence: {seq} -> Used by: {', '.join(table_details)}")
        print("--- End of Summary ---")
    else:
        print("\nNo sequences found to process after parsing HBM files.")


def update_sequences_in_db(db_username, db_password, db_dsn):
    """
    Connects to the Oracle database and updates sequences based on the
    maximum ID found in associated tables, using the INCREMENT BY method.
    """
    if not sequence_to_tables_map:
        print("No sequences to process. Exiting database update.")
        return

    connection = None
    cursor = None
    try:
        if ORACLE_CLIENT_LIB_DIR:
            try:
                print(f"Initializing Oracle Client from: {ORACLE_CLIENT_LIB_DIR}")
                cx_Oracle.init_oracle_client(lib_dir=ORACLE_CLIENT_LIB_DIR)
            except Exception as e_init:
                print(
                    f"Error initializing Oracle client: {e_init}. Ensure the path is correct and client is installed.")
                return

        print(f"\nAttempting to connect to Oracle Database with DSN: {db_dsn} and User: {db_username}...")
        connection = cx_Oracle.connect(user=db_username, password=db_password, dsn=db_dsn)
        cursor = connection.cursor()
        print("Successfully connected to Oracle Database.")

        for sequence_name, tables_using_sequence in sequence_to_tables_map.items():
            print(f"\nProcessing Sequence: '{sequence_name}'")
            overall_max_id = 0
            # max_id_found_for_sequence = False # To track if any table had a non-zero max_id

            for table_detail in tables_using_sequence:
                table_name = table_detail['table_name']
                pk_column_name = table_detail['pk_column']
                max_id_query = f"SELECT MAX({pk_column_name}) FROM {table_name}"
                try:
                    print(f"  Querying max ID for table '{table_name}' (PK: '{pk_column_name}')...")
                    cursor.execute(max_id_query)
                    result = cursor.fetchone()
                    current_table_max_id = result[0] if result and result[0] is not None else 0
                    # if result and result[0] is not None:
                    #     max_id_found_for_sequence = True
                    print(f"  Max ID in table '{table_name}' is: {current_table_max_id}")
                    if current_table_max_id > overall_max_id:
                        overall_max_id = current_table_max_id
                except cx_Oracle.DatabaseError as db_err_max_id:
                    err_obj_max_id, = db_err_max_id.args
                    print(
                        f"  Oracle Database Error querying max ID for table '{table_name}': Code={err_obj_max_id.code}, Message={err_obj_max_id.message.strip()}")
                    if err_obj_max_id.code == 942:
                        print(
                            f"  Hint: Table '{table_name}' might not exist or is not accessible by '{db_username}'. Skipping this table for max ID calculation of sequence '{sequence_name}'.")
                except Exception as e_max_id:
                    print(
                        f"  Unexpected error querying max ID for table '{table_name}': {e_max_id}. Skipping this table.")

            next_val_needed_for_insert = int(overall_max_id) + 1
            print(f"Overall Max ID for tables using sequence '{sequence_name}' is: {overall_max_id}")
            print(f"Required next value for successful insertion after max ID is: {next_val_needed_for_insert}")

            try:
                cursor.execute(
                    f"SELECT INCREMENT_BY FROM USER_SEQUENCES WHERE SEQUENCE_NAME = '{sequence_name.upper()}'")
                seq_info = cursor.fetchone()
                if not seq_info:
                    print(
                        f"Error: Sequence '{sequence_name}' not found in USER_SEQUENCES for user '{db_username}'. Skipping update for this sequence.")
                    continue
                original_increment_by = seq_info[0]
                print(f"Original INCREMENT_BY for '{sequence_name}' is: {original_increment_by}")

                # Step 1 (User's method): Get current nextval
                cursor.execute(f"SELECT {sequence_name}.NEXTVAL FROM DUAL")
                current_actual_nextval = cursor.fetchone()[0]
                print(f"Value obtained from first '{sequence_name}.NEXTVAL' call: {current_actual_nextval}")

                # Condition: If sequence's current nextval is already greater than overall_max_id,
                # it means it will provide overall_max_id + 1 or more, so it's fine.
                if current_actual_nextval > overall_max_id:
                    # More precisely: if current_actual_nextval >= next_val_needed_for_insert
                    if current_actual_nextval >= next_val_needed_for_insert:
                        print(
                            f"Sequence '{sequence_name}' (current nextval: {current_actual_nextval}) is already sufficient (>= {next_val_needed_for_insert}). No update needed for this sequence.")
                        connection.commit()  # Commit any DDL from previous sequences if any
                        continue  # Move to the next sequence
                    else:
                        # This case means current_actual_nextval is > overall_max_id but < next_val_needed_for_insert.
                        # This happens if overall_max_id = 0, next_val_needed_for_insert = 1, and current_actual_nextval = 0 (if sequence was just reset or minvalue is 0).
                        # However, Oracle sequences usually start at 1 or more.
                        # If current_actual_nextval = 0 (and overall_max_id = 0, next_val_needed = 1), we need to advance it to 1.
                        print(
                            f"Sequence '{sequence_name}' (current nextval: {current_actual_nextval}) needs adjustment to reach {next_val_needed_for_insert}.")

                # If we are here, sequence needs adjustment.
                # Target for the *next* .NEXTVAL call (after the one we just did) should be next_val_needed_for_insert.
                # The value we want the *second* .NEXTVAL (in our script) to return is next_val_needed_for_insert.
                jump_amount = next_val_needed_for_insert - current_actual_nextval
                print(
                    f"Calculated jump_amount for INCREMENT BY: {jump_amount} (Target: {next_val_needed_for_insert}, Current after 1st nextval: {current_actual_nextval})")

                # Step 2 (User's method): Alter increment by jump_amount
                # Only alter if jump_amount is not the same as original_increment_by,
                # or if it is, but current_actual_nextval wasn't already the target.
                # The crucial part is to make the *next* NEXTVAL call yield next_val_needed_for_insert.
                if jump_amount == 0 and current_actual_nextval == next_val_needed_for_insert:
                    print(
                        f"Sequence '{sequence_name}' first NEXTVAL was already the target {next_val_needed_for_insert}. Ensuring original increment is restored if it was changed by a failed previous run.")
                    # If jump_amount is 0, it means current_actual_nextval is already next_val_needed_for_insert.
                    # We still need to ensure the increment is set back to original_increment_by if it was altered.
                    if original_increment_by != 1:  # Assuming the alter below would have happened.
                        # This state implies we might not need the full alter-nextval-alter sequence,
                        # but to be safe and align with the user's pattern, we can proceed,
                        # knowing the jump_amount is 0.
                        pass  # Let the logic proceed, jump_amount being 0 is fine.

                alter_increment_command = f"ALTER SEQUENCE {sequence_name} INCREMENT BY {jump_amount}"
                print(f"Executing: {alter_increment_command}")
                cursor.execute(alter_increment_command)

                # Step 3 (User's method): Select nextval again
                cursor.execute(f"SELECT {sequence_name}.NEXTVAL FROM DUAL")
                adjusted_nextval = cursor.fetchone()[0]
                print(
                    f"Value after INCREMENT BY {jump_amount} and second '{sequence_name}.NEXTVAL': {adjusted_nextval}.")

                if adjusted_nextval != next_val_needed_for_insert:
                    print(
                        f"Warning: Sequence value {adjusted_nextval} after adjustment does not precisely match target {next_val_needed_for_insert}. Manual check strongly recommended.")
                    print(
                        f"  Debug info: overall_max_id={overall_max_id}, next_val_needed_for_insert={next_val_needed_for_insert}, current_actual_nextval_step1={current_actual_nextval}, jump_amount={jump_amount}")

                # Step 4 (User's method): Restore original increment by
                restore_increment_command = f"ALTER SEQUENCE {sequence_name} INCREMENT BY {original_increment_by}"
                print(f"Executing: {restore_increment_command}")
                cursor.execute(restore_increment_command)
                print(f"Sequence '{sequence_name}' increment restored to {original_increment_by}.")

                connection.commit()
                print(f"Sequence '{sequence_name}' successfully processed and committed.")

            except cx_Oracle.DatabaseError as db_err_seq_update:
                err_obj_seq, = db_err_seq_update.args
                print(
                    f"Oracle Database Error updating sequence '{sequence_name}': Code={err_obj_seq.code}, Message={err_obj_seq.message.strip()}")
                if err_obj_seq.code == 2289:  # ORA-02289: sequence does not exist
                    print(f"Hint: Ensure sequence '{sequence_name}' exists and is accessible by '{db_username}'.")
                if connection: connection.rollback()  # Rollback on error for this sequence
            except Exception as e_seq_update:
                print(f"Unexpected error updating sequence '{sequence_name}': {e_seq_update}")
                if connection: connection.rollback()

    except cx_Oracle.DatabaseError as db_err_conn:
        err_obj_conn, = db_err_conn.args
        print(
            f"Oracle Database connection/operation failed: Code={err_obj_conn.code}, Message={err_obj_conn.message.strip()}")
        if err_obj_conn.code == 1017:
            print("Hint: Check Oracle username/password.")
        elif err_obj_conn.code == 12541:
            print(f"Hint: Ensure Oracle TNS Listener is running on the host specified in DSN '{db_dsn}'.")
        elif err_obj_conn.code == 12514:
            print(f"Hint: Ensure service name in DSN '{db_dsn}' is correct.")
        elif err_obj_conn.code == 12154:
            print(f"Hint: Ensure DSN '{db_dsn}' is correctly formatted or defined in tnsnames.ora.")
    except Exception as e_main:
        print(f"An unexpected error occurred: {e_main}")
    finally:
        if cursor: cursor.close()
        if connection:
            connection.close()
            print("\nOracle Database connection closed.")


sequence_to_tables_map = {}  # 应该在 find_and_process_hibernate_configs 中被填充


def main(args):
    print("--- Oracle Sequence Updater for Hibernate HBM Files (INCREMENT BY Method) ---")

    hbm_dir = args.hbm_dir or input("Enter the root directory path for Hibernate .hbm.xml files: ").strip()
    if not os.path.isdir(hbm_dir):
        print(f"Error: The provided path '{hbm_dir}' is not a valid directory.")
        return

    print(f"\nScanning Hibernate HBM configurations in: {hbm_dir}")
    find_and_process_hibernate_configs(hbm_dir)

    if not sequence_to_tables_map:
        print("\nNo sequences to process based on HBM files.")
        return

    db_user = args.db_user or input("Enter Oracle DB Username: ").strip()
    db_password = args.db_password or input("Enter Oracle DB Password: ").strip()
    db_dsn = args.db_dsn or input("Enter Oracle DB DSN (e.g., 'host:port/service_name'): ").strip()

    if not (db_user and db_password and db_dsn):
        print("Database credentials and DSN are required. Exiting.")
        return

    update_sequences_in_db(db_user, db_password, db_dsn)
    print("\nScript finished.")


# --- Main Execution ---
if __name__ == "__main__":

    # ✅ 这里可以选择编码参数，也可以通过注释启用交互式
    use_code_args = True

    if use_code_args:
        args = argparse.Namespace(
            # hbm_dir="hbm_dir",
            # db_user="test",
            # db_password="test",
            # db_dsn="172.0.0.1:1521/orcl"
        )
    else:
        # 默认从命令行获取参数
        parser = argparse.ArgumentParser()
        parser.add_argument("--hbm-dir")
        parser.add_argument("--db-user")
        parser.add_argument("--db-password")
        parser.add_argument("--db-dsn")
        args = parser.parse_args()

    main(args)
