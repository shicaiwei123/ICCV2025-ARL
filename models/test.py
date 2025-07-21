def main():
    s = 'baca'
    res = 0
    for i in range(len(s)):
        tmp = 0
        for j in range(4 - i):
            tmp += 25 ** j
        res += (ord(s[i]) - ord('a')) * tmp
        res += 1
    print(res - 1)

main()
