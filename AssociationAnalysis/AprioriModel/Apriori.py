import numpy as np
from apyori import apriori


def aprioriModel(data,min_supp,min_conf):

    min_lift = 0.0
    res = apriori(transactions=data, min_support=min_supp, min_confidence=min_conf, min_lift=min_lift)

    resData = open("out/result.txt", 'w+')
    for rule in res:
        print(str(rule), file=resData)
    resData.close()

    print("ok")

# if __name__ == '__main__':
#     data = np.loadtxt("data/ShoppingData.txt", dtype=str)
#     min_supp = 0.5
#     min_conf = 0.8
#     min_lift = 0.0
#     res = apriori(transactions=data, min_support=min_supp, min_confidence=min_conf, min_lift=min_lift)
#
#     data = open("out/result.txt", 'w+')
#     for rule in res:
#         print(str(rule), file=data)
#     data.close()
#
#     print("ok")