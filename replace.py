for i in range(76):
    i_str = str(i)
    if i < 10:
        i_str = '0'+i_str

    with open('leaderboard/data/routes/route_'+i_str+'.xml', 'r') as f_in:
        with open('leaderboard/data/new_routes/route_'+i_str+'.xml', 'w') as f_out:
            s = f_in.read()
            s = s.replace('map', 'town')
            f_out.write(s)
