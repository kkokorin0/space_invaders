import os

if __name__ == "__main__":
    folder = r'C:\Users\kkokorin\Documents\GitHub\space_invaders\po_data'
    left_i = 0
    right_i = 0

    for file in os.listdir(folder):
        tokens = file.split('_')
        src_name = '%s//%s' % (folder, file)
        if int(tokens[0]) == 1:
            left_i += 1
            out_name = '%s//LEFT_%d.csv' % (folder, left_i)
        elif int(tokens[0]) == -1:
            right_i += 1
            out_name = '%s//RIGHT_%d.csv' % (folder, right_i)
        else:
            print(file)

        os.rename(src_name, out_name)
