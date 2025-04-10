import streamlit as st
# For myself:
# Sufficient: understanding of business needs and industry context by developing project proposal that
# effectively identifies and addressses client requirements. 
# Clear problem statement, define project objectives, and propose feasible solutions that demonstrate
# understanding of business environment.

# Good: understanding business context in final presentation, communicate insights. 
# engaging, well-structured, emphasizes practical implications within broader business context.

# Excellent: innovative solution that not only aligns with client's specific requirements,
# but also possesses considerable business value


def main():
    
    
    st.title('Safer Roads in Breda')
    st.divider()
    st.subheader('Block 1D 2023-2024 Project')
    st.subheader('By: Lars Kerkhofs, Natalia Mikes, Artjom Musaelans, Dominik Ptaszek and Luka Wieme')
    st.divider()
    st.title('The Client and their Issue')
    st.write(''' 
    Imagine you are driving down an urban road in Breda, just minding your own business. However,
             due to unforeseen circumstances, for example, overgrown bushes, sidestreets that are hidden
             from view or hard rain, you are unable to see the oncoming intersection.
             This could lead to a dangerous situation or potentially a traffic incident. Too many incidents
             happen on rural roads especially. Our client, ANWB, came to us with this rising issue and
             asked us what could be done about it.  
             Looking at the Netherlands overall, it is one of 
             the best countries at enforcing safer traffic for everyone. Many programmes have been 
             implemented over the years to improve traffic safety, all under the name Sustainable Safety
             (SWOV, 2018). Three principles were followed when implementing measures to improve road
             safety: eliminating, minimizing and mitigating. We decided to focus on mitigating effects
             with the data that was available to us. The mitigating principle holds in that 'where people
             are exposed to risks, their consequences should as far as possible be mitigated by taking 
             appropriate mitigating measures' (SWOV, 2018, p. 8). Thus, our project not only aligns
             well with the client's needs, but also on a national level.  
             The purpose of the project would be to mitigate risk on urban roads in Breda. It was 
             noticed that tons of applications exist that show traffic information on a broad scale,
             not on a more local scale, like per street. Even if a street is not busy, it could still
             be potentially dangerous to drive through due to a myriad of factors. Our application takes
             into account these factors and labels streets in Breda as either low-risk or high-risk. 
    ''')
    st.markdown(''' 
    :triangular_flag_on_post: :collision: :red_car: 
                ''')
    st.title('Project Idea')
    st.write('''
    Our project combines available datasets, including the ANWB safe driving dataset, KNMI weather datasets
             and BRON accident dataset to determine whether a street is high-risk or low-risk to drive
             through. With the help of this application, users (mostly drivers, but also other traffic
             users) will be more prepared. Users could check ahead of time what kind of streets they 
             will be driving through and depending on that could decide to reroute or be more aware
             when driving through streets labelled high-risk. Users will be more alert and drive more safely
             when using this application, which will lead to less incidents.  
             This will also be beneficial
             for ANWB's stakeholders to lower the costs of insurance. It aligns well with one of ANWB's missions
             as stated on their website, which is to prevent rather than insure. ANWB can also use this 
             application to identify high-risk streets and confront the municipality of Breda to implement
             more safety measures in those areas. 
             ''')


if __name__ == '__main__':
    main()