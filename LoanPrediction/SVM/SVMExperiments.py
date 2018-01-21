from SVM import SVM

def test_parameters(features, target, kernel="linear", Cs=[1], gammas=[0]):

    best_param = (0, [])
    for c in Cs:
        print("C: " + str(c))
        for g in gammas:
            print("g: " + str(g))
            model, acc, _ = SVM().work(features, target, kernel, c, g)
            print("acc: "+str(acc))
            if best_param[0] < acc:
                best_param = (acc, [kernel, c, g])

    print("Best params: " + str(best_param[0]) +"-"+ str(best_param[1]))