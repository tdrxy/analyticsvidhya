def solve(model, testdata, ids=[], outfile="solutions.csv"):
    predictions = model.predict(testdata)

    write_seq = ["Loan_ID,Loan_Status\n"]
    for i, prediction in enumerate(predictions):
        line = str(ids.values[i]+","+_map_prediction(prediction))+"\n"
        write_seq.append(line)

    with open(outfile, 'w+') as fh:
        fh.writelines(write_seq)
        fh.close()

def _map_prediction(prediction):
    if prediction[0] == 1:
        return "N"
    else:
        return "Y"
