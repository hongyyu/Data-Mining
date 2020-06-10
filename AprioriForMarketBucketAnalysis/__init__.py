# def output_triple_etc(file, ans):
#     """Output both pairs and triples file in correct format
#     :parameter file: output of ans after transforming
#     :type file: str
#     :parameter ans: answer of pairs or triples
#     :type ans: list
#     :return file
#     :rtype file: str
#     """
#
#     for i in range(len(ans)):
#         if i == len(ans) - 1:
#             file += str(ans[i]) + '\n'
#         else:
#             file += str(ans[i]) + ','
#     return file