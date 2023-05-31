import pickle
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from pickle_vect import deserialize_DecisionTree, tfidf_Deserialize, deserialize_dict_vect, deserialize_encode_state
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk import word_tokenize
import scipy.sparse

from tokenize_json import tfidf_vectorize

def parse_pronto(title,description,feature,build):

    text_vectorizer = tfidf_Deserialize()
    porter = nltk.PorterStemmer()
    twd = TreebankWordDetokenizer()
    tokens = word_tokenize(title + ' ' + description)
    stemmed_tokens = [porter.stem(t) for t in tokens]
    text = TreebankWordDetokenizer.detokenize(twd, stemmed_tokens)
    text_to_matrix = text_vectorizer.transform([text])


    feature_vectorize = deserialize_dict_vect()
    print(feature_vectorize)
    feature_dict = {'build' : build , 'feature' : feature}
    print(feature_dict)
    feature_matrix = feature_vectorize.transform(feature_dict)
    print("Printing the feature_matrix ", feature_matrix)

    data_matrix = scipy.sparse.hstack((feature_matrix, text_to_matrix))
    print("Printing the data_matrix",data_matrix)
    return data_matrix


def decode_state(label):
    encoding = LabelEncoder()
    encoding.fit(deserialize_encode_state())

    label_encoder = encoding.inverse_transform(label)
    print("Printing the label_encoder", label_encoder)
    return label_encoder[0]

def predict(data_matrix):
    clf = deserialize_DecisionTree()

    label_result = clf.predict(data_matrix)
    print("Printing the label_result", label_result)
    return decode_state(label_result)   # un numar - trebuie decodat cu label encoder , decode()

if __name__ == "__main__":

    title = "[ST][FDD][RFS][CNI80290][AIRSCALE+AIRSCALE][SBTS22R2][ASIL+ASIA][ABIO][AHFII][CPRI]FID10 reported on 4G RFs while PL after 5G config reset SL"
    description = "**** DEFAULT TEMPLATE for 2G-3G-4G-5G-SRAN-FDD-TDD-DCM-Micro-Controller common template v1.4.0 (02.07.2021) – PLEASE FILL IT IN BEFORE CREATING A PR AND DO NOT CHANGE / REMOVE ANY SECTION OF THIS TEMPLATE ****\r\n\r\n[1. Detail Test Steps:]\r\n1. 4G up&running with primary link ; 5G up&running with secondary link\r\n2. Backup commissioning file (SCF) is saved from 5G master technologly .Configuration reset is done on 5G\r\n3. Planned commisioning is started on 5G\r\n[2. Expected Result:]\r\n1.4G up&running with primary link ; 5G up&running with secondary link\r\n2.OK,4G is not affected\r\n3.both BTSs are operational\r\n[3. Actual Result:]\r\n1.OK\r\n2.NOK : FID 10 is reported on 4G side while having primary link after 5G configuration reset\r\n3.OK\r\n[4. Tester analysis:]\r\nSYSLOG_75\r\n070680 15.03 13:58:33.356 [192.168.255.129] b4 FCT-1011-3-FRI 2022-03-15T11:58:33.346803Z 34FE-FRI INF/FRI/FRI, [Correlator 10 /MRBTS-1/RAT-1/RUNTIME_VIEW-1/MRBTS_R-1/EQM_R-1/APEQM_R-1/RMOD_R-1] active\r\n070920 15.03 13:58:33.992 [192.168.255.129] f6 FCT-1011-3-FRI 2022-03-15T11:58:33.991199Z 34FE-FRI INF/FRI/FRI, [Correlator 10 /MRBTS-1/RAT-1/RUNTIME_VIEW-1/MRBTS_R-1/EQM_R-1/APEQM_R-1/RMOD_R-3] active\r\n071222 15.03 13:58:35.065 [192.168.255.129] 79 FCT-1011-3-FRI 2022-03-15T11:58:35.044363Z 34FE-FRI INF/FRI/FRI, [Correlator 10 /MRBTS-1/RAT-1/RUNTIME_VIEW-1/MRBTS_R-1/EQM_R-1/APEQM_R-1/RMOD_R-2] active\r\n095308 15.03 13:59:49.332 [192.168.255.129] a3 FCT-1011-3-FRI 2022-03-15T11:59:49.213411Z 34FE-FRI INF/FRI/FRI, [Correlator 10 /MRBTS-1/RAT-1/RUNTIME_VIEW-1/MRBTS_R-1/EQM_R-1/APEQM_R-1/RMOD_R-1] inactive\r\n098932 15.03 13:59:49.717 [192.168.255.129] 3d FCT-1011-3-FRI 2022-03-15T11:59:49.677744Z 34FE-FRI INF/FRI/FRI, [Correlator 10 /MRBTS-1/RAT-1/RUNTIME_VIEW-1/MRBTS_R-1/EQM_R-1/APEQM_R-1/RMOD_R-3] inactive\r\n103003 15.03 13:59:50.448 [192.168.255.129] a5 FCT-1011-3-FRI 2022-03-15T11:59:50.329747Z 34FE-FRI INF/FRI/FRI, [Correlator 10 /MRBTS-1/RAT-1/RUNTIME_VIEW-1/MRBTS_R-1/EQM_R-1/APEQM_R-1/RMOD_R-2] inactive\r\nNIDD:\r\nReporting method to O&M\t1a.BTS SC: BTSOM SW(FR/FHS case)# discovers A. that communication with FR/FHS on RP1 level is lost (architecture with RP1 FR). or B. no heartbeat status or internal alarm received from radio (architecture with SNMP and CPRI-A radios). 1b.BTS SC: BTSOM SW (RET, LNA case)# discovers that communication with RET or LNA on HDLC level is lost. 1c.BTS SC: BTSOM SW (FSP, FBB, ABIx case)# discovers that DSP doesn't reply to HWAPI message. 2.BTS SC: BTSOM SW# changes states in InfoModel. 3.BTS SC: BTSOM SW (SOAM_FRI)# detects fault situation based on changes in InfoModel object states.\r\n\r\n[5. Pronto Creator Interface (PCI) information:]\r\n\r\nPCI executed for Problem Report (15.03.2022, 13:18:56 UTC):\r\nPCI ID: PCI_20220315_141806_000\r\nMLoGIC out of order\r\nAccess PCI to check saved results: https://rep-portal.wroclaw.nsn-rdnet.net/pci/?pci_id=PCI_20220315_141806_000\r\n\r\n[6. Log(s) file name containing a fault: (clear indication (exact file name) and timestamp where fault can be found in attached logs):]\r\n\\\\eseefsn50.emea.nsn-net.net\\rotta4internal\\5G_2\\AdinaHotac\\newPR_AHFII\\4g\r\n[7. Test-Line Reference/used HW/configuration/tools/SW version:]\r\n5G+LTE+3xAHFII RFS\r\n5G (ASIL + ABIO): SBTS22R2_ENB_0000_000962_000000\r\nLTE (ASIA + ABIA): SBTS22R2_ENB_0000_000962_000000\r\n[8. Used Flags: (list here used R&D flags):]\r\n#0x1003f=1 #ERadCcs_AaSysLogInputLevel - Severity changed from DEBUG to INF\r\n#0x10040=1 #ERadCcs_AaSysLogOutputLevel - Severity changed from DEBUG to INFO\r\n0x10041=5 #ERadCcs_AaSysLogInputLevel - Severity changed from DEBUG to INF\r\n0x10042 = 0xC0A8ff82  #for VMs with debug interface 192.168.255.130; for IP ending in .126 we have 7E instead of 82\r\n0x10043 = 0xc738\r\n#PR_RFM_Block_Reset_WebEM\r\n0x1003F=1 \r\n0x10040=1\r\n245=2  # FEAT_DBG_Rpmag\r\n250=6000000  #FEAT_DBG_BTSOM_Enable_SoapTrace\r\n252=2  #FEAT_DBG_BTSOM_APW\r\n#ETH Security disabled\r\n0x1A0020=1\r\n#IM Gateway needed for IM Commander, enables port 12345; can be removed for CommonTrunk and FL18SP releases\r\n0x3A0001 = 1\r\n#Enabling ADMIN developer account:\r\n0x45005A = 1\r\n\r\n#BBC debug\r\n275=1\r\n0x42000f = 1 #DCS Timings Debug\r\n0x13001E = 1 #UEC TUecRadDebugLogEnabledCommon\r\n0x13001F = 1 #UEC TUecRadDebugLogEnabledCommonCodec\r\n0x130020 = 1 #UEC TUecRadDebugLogEnabledGlobal\r\n0x130021 = 1 #UEC TUecRadDebugLogEnabledUec\r\n#Maximum number of resets allowed within one hour\r\n0x310001 = 300\r\n#FSP recovery resets allowed within one hour\r\n0x500002 = 30\r\n#LTE recovery resets allowed within one hour\r\n0x500001 = 30\r\n#Fallback related flags, disables recovery in case of process crash \r\n# fallback protection (no recovery in case of process crash)\r\n# Disable Watchdog\r\n# 0x10077=1\r\n# set core dump size\r\n# 0x10079=200000\r\n# 0x300F=1\r\n#245=2\r\n0x190030=0  #HW files\r\n#0x300020=30\r\n0x4F00D0=0\r\n#TCP DUMP:\r\n0x3700E2=1\r\n0x3700E3=15\r\n[9. Test Scenario History of Execution: (what was changed since it was tested successfully for the last time):]\r\n\r\nWas Test Scenario passing before? ( YES | NO | New scenario ) New scenario\r\n\r\nWhat was the last SW version Test Scenario was passing? ( SW load | New scenario ) New scenario\r\n\r\nWere there any differences between test-lines since last time Test Scenario was passing? ( YES, explanation | NO | New test-line ) New test-line\r\n\r\nWere there any changes in Test Scenario since last run it passed? ( YES, explanation | NO | New scenario ) New scenario\r\n\r\n[10. Test Case Reference: (QC, RP or UTE link):]\r\n\r\nTest Instance ID: 254503 [1]5G_RFS_BTS_Commissioning_Green_Field_Installation\r\n**** END OF DEFAULT TEMPLATE ****"
    feature = "CNI-80290-A-B1"
    build = "SBTS22R2_ENB_0000_000962_000000"

    parse = parse_pronto(title,description,feature,build)
    
    prediction = predict(parse)
    prediction = decode_state(predict(parse))
    print("Prediction: " , prediction)

