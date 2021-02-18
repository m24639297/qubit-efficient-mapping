def get_num_groups(qubitOp):
    group_list = []
    for operator in qubitOp._paulis_table.keys():
        # I, Z = 0
        # X = 1
        # Y = 2
        operator_in_num = ''
        for matrix in operator:
            if matrix == 'I' or matrix == 'Z':
                operator_in_num += '0'
            elif matrix == 'X':
                operator_in_num += '1'
            elif matrix == 'Y':
                operator_in_num += '2'
        group_list.append(operator_in_num)
    group_list = list(dict.fromkeys(group_list))
    return len(group_list)
