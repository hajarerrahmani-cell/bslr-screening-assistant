import io, hashlib
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

APP_VERSION='1.0.0'

def norm(x):
    return str(x).strip().lower().replace(' ','_').replace('-','_').replace('/','_')

def load_table(f):
    n=f.name.lower()
    if n.endswith('.csv'): return pd.read_csv(f)
    if n.endswith(('.xlsx','.xls')): return pd.read_excel(f)
    raise ValueError('استخدمي CSV أو Excel')

def find_col(df,cands):
    m={norm(c):c for c in df.columns}
    for x in cands:
        if x in m: return m[x]
    for k,v in m.items():
        if any(x in k for x in cands): return v
    return None

def infer(df):
    return {
        'title':find_col(df,['title','titre','ti']),
        'abstract':find_col(df,['abstract','resume','résumé','ab']),
        'keywords':find_col(df,['keywords','keyword','mots_cles','de']),
        'doi':find_col(df,['doi','di']),
        'authors':find_col(df,['authors','auteurs','au']),
        'year':find_col(df,['year','annee','année','py'])}

def series(df,col):
    return pd.Series(['']*len(df),index=df.index) if not col else df[col].fillna('').astype(str)

def compose(df,m):
    return ('TITLE: '+series(df,m['title'])+' ABSTRACT: '+series(df,m['abstract'])+' KEYWORDS: '+series(df,m['keywords'])).str.replace(r'\s+',' ',regex=True).str.strip()

def rid(row,m,idx):
    doi=str(row[m['doi']]) if m['doi'] and pd.notna(row[m['doi']]) else ''
    title=str(row[m['title']]) if m['title'] and pd.notna(row[m['title']]) else ''
    raw=(doi.strip().lower() or title.strip().lower() or str(idx)).encode()
    return hashlib.md5(raw).hexdigest()[:12]

def prep_examples(df,label):
    if df is None or df.empty: return pd.DataFrame(columns=['text','label'])
    m=infer(df); out=pd.DataFrame({'text':compose(df,m),'label':label})
    return out[out.text.str.len()>10]

def evidence(text):
    t=str(text).lower()
    pos_terms=['sectoral stock','sector stock','sector index','sectoral index','industry index','industry portfolio','sectoral portfolio','sectoral equity','equity sector','volatility','spillover','connectedness','contagion','systemic risk','tail risk','value at risk','expected shortfall','market risk','resilience','stability','beta']
    neg_terms=['public sector','health sector','agricultural sector','education sector','government sector','firm-level','firm level','operational risk','fraud detection']
    p=[x for x in pos_terms if x in t][:6]; n=[x for x in neg_terms if x in t][:6]
    return ', '.join(p) or '—', ', '.join(n) or '—'

def xlsx_bytes(df,audit):
    b=io.BytesIO()
    with pd.ExcelWriter(b,engine='openpyxl') as w:
        df.to_excel(w,index=False,sheet_name='SCREENING_RESULTS')
        audit.to_excel(w,index=False,sheet_name='AUDIT_LOG')
    b.seek(0); return b.getvalue()

st.set_page_config(page_title='BSLR Screening Assistant',page_icon='📚',layout='wide')
st.title('📚 BSLR Screening Assistant')
st.caption('Screening académique assisté: règles + TF-IDF + Logistic Regression + validation humaine')

with st.sidebar:
    project=st.text_input('Nom du projet','Sectoral Market Risk Review')
    st.info("النموذج يساعد على ترتيب المقالات. القرار النهائي INCLUDE/EXCLUDE يبقى للباحث.")

T1,T2,T3,T4=st.tabs(['1️⃣ البيانات والمعايير','2️⃣ تدريب النموذج','3️⃣ النتائج والمراجعة','4️⃣ التصدير والتدقيق'])

with T1:
    main_file=st.file_uploader('الملف الرئيسي CSV/Excel',type=['csv','xlsx','xls'],key='main')
    c1,c2=st.columns(2)
    with c1: pos_file=st.file_uploader('أمثلة Relevant',type=['csv','xlsx','xls'],key='pos')
    with c2: neg_file=st.file_uploader('أمثلة Not relevant',type=['csv','xlsx','xls'],key='neg')
    rq=st.text_area('سؤال البحث','Market risk and stability of sectoral stock indices.',height=80)
    criteria=st.text_area('معايير الإدراج والاستبعاد','''INCLUDE:\n- sectoral stock/equity indices\n- industry indices or sectoral portfolios\n- volatility, market risk, systemic/tail risk\n- VaR / Expected Shortfall\n- spillovers, connectedness, contagion\n- stability, resilience, beta\n\nEXCLUDE:\n- firm-level only without sectoral index/portfolio\n- public/health/agriculture/education sectors unrelated to equity markets\n- operational/fraud/pure credit-risk studies without equity-market analysis\n- non-financial sector studies outside stock/equity markets''',height=260)
    if main_file:
        df=load_table(main_file); m=infer(df)
        st.success(f'{len(df):,} références chargées')
        opts=['—']+list(df.columns)
        def ix(c): return opts.index(c) if c in opts else 0
        a,b,c,d=st.columns(4)
        with a: m['title']=st.selectbox('Title',opts,index=ix(m['title']))
        with b: m['abstract']=st.selectbox('Abstract',opts,index=ix(m['abstract']))
        with c: m['keywords']=st.selectbox('Keywords',opts,index=ix(m['keywords']))
        with d: m['doi']=st.selectbox('DOI',opts,index=ix(m['doi']))
        for k in ['title','abstract','keywords','doi']:
            if m[k]=='—': m[k]=None
        work=df.copy(); work['record_id']=[rid(r,m,i) for i,r in work.iterrows()]
        work['text_for_model']=compose(work,m); work['model_probability']=np.nan; work['ai_category']=''; work['human_decision']=''; work['exclusion_reason']=''; work['review_note']=''
        st.session_state.update(main_df=df,screening_df=work,mapping=m,research_question=rq,criteria=criteria)
        st.dataframe(df.head(10),use_container_width=True)

with T2:
    if 'screening_df' not in st.session_state:
        st.warning('ارفعي الملف الرئيسي أولاً')
    else:
        pos=load_table(pos_file) if pos_file else pd.DataFrame(); neg=load_table(neg_file) if neg_file else pd.DataFrame()
        train=pd.concat([prep_examples(pos,1),prep_examples(neg,0)],ignore_index=True)
        if 'screening_df' in st.session_state:
            s=st.session_state.screening_df
            dec=s[s.human_decision.isin(['INCLUDE','EXCLUDE'])]
            if len(dec):
                z=pd.DataFrame({'text':dec.text_for_model,'label':(dec.human_decision=='INCLUDE').astype(int)})
                train=pd.concat([train,z],ignore_index=True)
        train=train.drop_duplicates()
        pc=int((train.label==1).sum()) if len(train) else 0; nc=int((train.label==0).sum()) if len(train) else 0
        x,y,z=st.columns(3); x.metric('Relevant',pc); y.metric('Not relevant',nc); z.metric('Total training',len(train))
        maxf=st.slider('TF-IDF max features',3000,30000,12000,1000)
        if st.button('🚀 Train + Score',type='primary'):
            if pc<5 or nc<5: st.error('نحتاج على الأقل 5 أمثلة من كل فئة')
            else:
                X=train.text.astype(str); Y=train.label.astype(int)
                pipe=Pipeline([('tfidf',TfidfVectorizer(stop_words='english',ngram_range=(1,2),max_features=maxf)),('clf',LogisticRegression(max_iter=2000,class_weight='balanced',random_state=42))])
                metrics={}
                if len(train)>=30 and Y.value_counts().min()>=5:
                    Xt,Xv,yt,yv=train_test_split(X,Y,test_size=.25,random_state=42,stratify=Y); pipe.fit(Xt,yt); pred=pipe.predict(Xv)
                    p,r,f,_=precision_recall_fscore_support(yv,pred,average='binary',zero_division=0); metrics={'precision':float(p),'recall':float(r),'f1':float(f),'test_n':len(yv)}
                pipe.fit(X,Y); sc=st.session_state.screening_df.copy(); sc['model_probability']=pipe.predict_proba(sc.text_for_model.astype(str))[:,1]
                st.session_state.update(model=pipe,metrics=metrics,screening_df=sc); st.success('تم التدريب والتقييم')
        if st.session_state.get('metrics'): st.json(st.session_state.metrics)

with T3:
    if 'screening_df' not in st.session_state or st.session_state.screening_df.model_probability.isna().all():
        st.warning('درّبي النموذج أولاً')
    else:
        low=st.slider('LOW threshold',0.0,.5,.25,.01); high=st.slider('HIGH threshold',.5,1.0,.80,.01)
        df=st.session_state.screening_df.copy()
        df['ai_category']=df.model_probability.apply(lambda p:'HIGH RELEVANCE' if p>=high else ('LOW RELEVANCE' if p<=low else 'MANUAL REVIEW'))
        ev=[evidence(t) for t in df.text_for_model]; df['positive_evidence']=[x[0] for x in ev]; df['negative_evidence']=[x[1] for x in ev]
        st.session_state.screening_df=df
        a,b,c=st.columns(3); a.metric('HIGH',int((df.ai_category=='HIGH RELEVANCE').sum())); b.metric('MANUAL REVIEW',int((df.ai_category=='MANUAL REVIEW').sum())); c.metric('LOW',int((df.ai_category=='LOW RELEVANCE').sum()))
        mode=st.selectbox('المجموعة للمراجعة',['MANUAL REVIEW','HIGH RELEVANCE','LOW RELEVANCE','كل غير المحسوم'])
        pending=df[df.human_decision==''].copy();
        if mode!='كل غير المحسوم': pending=pending[pending.ai_category==mode]
        pending['uncertainty']=(pending.model_probability-.5).abs(); pending=pending.sort_values('uncertainty' if mode=='MANUAL REVIEW' else 'model_probability',ascending=(mode=='MANUAL REVIEW'))
        if len(pending):
            row=pending.iloc[0]; idx=row.name; m=st.session_state.mapping
            title=str(row[m['title']]) if m['title'] else '(No title)'; abstract=str(row[m['abstract']]) if m['abstract'] else ''; kw=str(row[m['keywords']]) if m['keywords'] else ''; doi=str(row[m['doi']]) if m['doi'] else ''
            st.markdown('### '+title); st.progress(float(row.model_probability)); st.write(f"**Score:** {row.model_probability:.1%} — **{row.ai_category}**")
            if doi: st.write('**DOI:** '+doi)
            st.write('**Abstract**'); st.write(abstract or '—'); st.write('**Keywords**'); st.write(kw or '—'); st.write('**Positive evidence:** '+row.positive_evidence); st.write('**Negative evidence:** '+row.negative_evidence)
            reason=st.selectbox('سبب الاستبعاد',['','Non-equity / non-stock-market','Firm-level only','Wrong sector meaning','Wrong outcome','Wrong document / scope','Other']); note=st.text_input('ملاحظة')
            b1,b2,b3=st.columns(3)
            if b1.button('✅ INCLUDE',use_container_width=True): st.session_state.screening_df.loc[idx,['human_decision','review_note']]=['INCLUDE',note]; st.rerun()
            if b2.button('❌ EXCLUDE',use_container_width=True): st.session_state.screening_df.loc[idx,['human_decision','exclusion_reason','review_note']]=['EXCLUDE',reason,note]; st.rerun()
            if b3.button('❓ UNCERTAIN',use_container_width=True): st.session_state.screening_df.loc[idx,['human_decision','review_note']]=['UNCERTAIN',note]; st.rerun()
        else: st.success('لا توجد مقالات متبقية في هذه الفئة')
        show=[x for x in ['record_id',st.session_state.mapping['title'],st.session_state.mapping['doi'],'model_probability','ai_category','human_decision','exclusion_reason'] if x]
        st.dataframe(df[show].sort_values('model_probability',ascending=False),use_container_width=True,height=420)

with T4:
    if 'screening_df' not in st.session_state: st.warning('لا توجد نتائج')
    else:
        df=st.session_state.screening_df.copy(); audit=pd.DataFrame([{'timestamp_utc':datetime.utcnow().isoformat()+'Z','app_version':APP_VERSION,'project_name':project,'research_question':st.session_state.get('research_question',''),'eligibility_criteria':st.session_state.get('criteria',''),'n_records':len(df),'n_include':int((df.human_decision=='INCLUDE').sum()),'n_exclude':int((df.human_decision=='EXCLUDE').sum()),'n_uncertain':int((df.human_decision=='UNCERTAIN').sum()),'random_state':42,'model':'TF-IDF + Logistic Regression'}])
        st.download_button('⬇️ CSV',df.to_csv(index=False).encode('utf-8-sig'),'screening_results.csv','text/csv')
        st.download_button('⬇️ Excel + Audit',xlsx_bytes(df,audit),'screening_results_with_audit.xlsx','application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
