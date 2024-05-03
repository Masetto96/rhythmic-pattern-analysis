# rythmic-pattern-analysis
## internship @ music technology group

Project Overview: 

Different representations have been proposed in the MIR literature to characterize rhythm patterns within music. These can be used to infer tempo, meter, and small-scale deviations (e.g., swing) from an audio signal but also serve as a preprocessing step in many tasks that depend on similarity, including genre classification [1] and music retrieval [2]. The recognition of rhythm patterns from audio has received much less attention than other rhythm analysis tasks, such as tempo estimation or beat tracking [3]. The technique described in [4] exploits the properties of the scale transform to achieve a rhythm descriptor that is robust to tempo variations. This was later extended by combining it with the modulation spectrum and adding correlation coefficients between frequency bands [5, 6], yielding the state of the art in rhythm pattern recognition. In recent work, a representation of the harmonic series at the tempo frequency is used as input to a convolutional network, which is trained to estimate rhythm pattern classes [7]. It produces promising results but falls behind the state of the art. In this thesis project, we will explore the existing approaches for rhythm pattern recognition applied to a wider variety of music genres than those used in previous studies. 

[1] Foote, J., Cooper, M. L. and Nam, U. (2002) Audio retrieval by rhythmic similarity. In Proc. of International Society for Music Information Retrieval (ISMIR)

[1] Tzanetakis, G. and Cook, P. (2002) Musical genre classification of audio signals. IEEE Transactions on Speech and Audio Processing.

[2] Foote, J., Cooper, M. L. and Nam, U. (2002) Audio retrieval by rhythmic similarity. In Proc. of International Society for Music Information Retrieval (ISMIR)

[3] Geoffroy Peeters, G. (2011) Spectral and temporal periodicity representations of rhythm for the automatic classification of music audio signal. IEEE Transactions on Audio, Speech, and Language Processing.

[4] Holzapfel, A. and Stylianou, Y. (2011). Scale transform in rhythmic similarity of music. IEEE Transactions on Audio, Speech, and Language Processing.

[5] Marchand, U. and Peeters, G. (2014). The modulation scale spectrum and its application to rhythm-content description. In Proc. of International Conference on Digital Audio Effects (DAFx)

[6] Marchand, U. and Peeters, G.. (2016). Scale and shift invariant time/frequency representation using auditory statistics: Application to rhythm description. In IEEE 26th International Workshop on Machine Learning for Signal Processing (MLSP)

[7] Foroughmand, H., and Peeters, G. (2019). Deep-rhythm for tempo estimation and rhythm pattern recognition. In *International Society for Music Information Retrieval (ISMIR)*.

#### The steps to carry out for this project are:
- Implement some rhythm descriptors (that will be integrated into Essentia).
- Compare existing rhythm descriptors for the task of rhythm pattern recognition.
- Ideally, propose improvements for the deep learning approach to reach the state of the art.