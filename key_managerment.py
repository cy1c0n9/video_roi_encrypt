import key
import sys

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('error: system argument')
    else:
        op = sys.argv[1]
        fid = sys.argv[2]
        if op == 'n':
            key.generate_new_key(fid)
