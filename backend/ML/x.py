names = ["ISIC_0942654.dcm",  "ISIC_1384614.dcm",  "ISIC_1629539.dcm",  "ISIC_1927858.dcm",  "ISIC_2131585.dcm",  "ISIC_2452083.dcm",  "ISIC_2899829.dcm",  "ISIC_3438779.dcm",  "ISIC_5624660.dcm",  "ISIC_5781547.dcm",  "ISIC_5919794.dcm",  "ISIC_6041880.dcm",  "ISIC_6447132.dcm",  "ISIC_7189099.dcm",  "ISIC_7537925.dcm",  "ISIC_7646167.dcm",  "ISIC_8125684.dcm",  "ISIC_8580217.dcm",  "ISIC_8922111.dcm",  "ISIC_9616986.dcm"]

target_0=0
target_1=0
with open("train.csv") as f:
    for x in f:
        ans = x.split(',')
        #print(ans)
        if ans[0]+'.dcm' in names and ans[-1].strip() == "0":
            target_0+=1
        elif ans[0]+'.dcm' in names and ans[-1].strip() == "1":
            target_1+=1
print(target_0, target_1)




