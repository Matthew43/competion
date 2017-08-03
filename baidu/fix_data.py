val_id_map_label_file = "E:/python_study/data/val.txt"
train_id_map_label_file = 'E:/python_study/data/data_train_image.txt'


def view():
    vlist = []
    tlist = []
    with open(val_id_map_label_file, 'r') as f:
        for l in f.readlines():
            vlist.append(int(l.split(' ')[1]))
    with open(train_id_map_label_file, 'r') as f:
        for l in f.readlines():
            tlist.append(int(l.split(' ')[1]))

    s = set([i for i in range(134)])
    s_100 = set([i for i in range(100)])
    v_set = set(vlist)
    t_set = set(tlist)
    print(len(v_set), ' ', len(t_set))
    print('v和t: ', v_set - t_set, ' and ', t_set - v_set)
    print('差：', t_set - s_100)
    print('差：', v_set - s_100)
    print('差：', s_100 - v_set)
    you = v_set - s_100
    que = s_100 - v_set
    print(len(you), ' ', len(que))
    with open('fix_map.txt','w') as f:

        for y, q in zip(you, que):
            print(y, ' ', q)
            # f.write(str(y)+' '+ str(q)+'\n')

def fix_data():
    fix_map = {}
    with open('fix_map.txt', 'r') as f:
        for l in f.readlines():
            t = l.split(' ')
            fix_map[t[0]] = t[1].strip('\n')

    vlist = []
    tlist = []
    with open(val_id_map_label_file, 'r') as f:
        for l in f.readlines():
            vlist.append(l.split(' '))
    with open(train_id_map_label_file, 'r') as f:
        for l in f.readlines():
            tlist.append(l.split(' '))

    print(len(tlist))
    print(len(vlist))

    with open('new_train.txt', 'w') as f:
        for l in tlist:
            if l[1] in fix_map:
                # print(l[1],'-->',fix_map[l[1]])
                l[1] = fix_map[l[1]]
            f.write(l[0]+' '+l[1]+'\n')


    with open('new_val.txt', 'w') as f:
        for l in vlist:
            if l[1] in fix_map:
                # print(l[1],'-->',fix_map[l[1]])
                l[1] = fix_map[l[1]]
            f.write(l[0]+' '+l[1]+'\n')




if __name__ == '__main__':
    # view()
    fix_data()
