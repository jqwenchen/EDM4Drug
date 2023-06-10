# 12
#
# C       19.51600       14.54700       13.49400
# C       19.29600       15.68700       14.23800
# H       18.77340       15.62000       15.18530
# C       19.72400       16.87500       13.79700
# H       19.56810       17.76250       14.39960
# C       20.35800       16.97500       12.58800
# H       20.69480       17.94310       12.23520
# C       20.56200       15.90100       11.84900
# H       21.04700       15.99740       10.88430
# C       20.16600       14.66300       12.29000
# H       20.36530       13.78170       11.69110
# H       19.18030       13.57230       13.85480


def read_one_xyz(filename):
    xyz = []
    with open(filename, 'r') as f:
        content = f.read()
        contact = content.split('\n')
        cur_file_name = ""
        for line in contact:
            line = line.strip()
            if line == '' or line.isdigit():
                continue
            if line.endswith("cdx") or "maegz" in line:
                cur_file_name = line
                continue
            with open(f"/home/p1/Desktop/data/xyz/{cur_file_name}", "a+") as f:
                f.write(line)
                f.write("\n")
            atom = line.split()
            xyzitem = {'symbol': atom[0], 'position': {"x": atom[1], "y": atom[2], "z": atom[3]}}
            xyz.append(xyzitem)
    return xyz


if __name__ == '__main__':
    xyz = read_one_xyz('/home/p1/Desktop/data/230605_FDA_Compound_Library.xyz')
    for f in xyz:
        print(f)

