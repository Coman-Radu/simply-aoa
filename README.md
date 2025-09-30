# Multi-source Sound Source Localization

We are using a microphone array with equal spacing between each microphone along the cardinal directions. Using time delay to determine the angle of arrival along both axes on the array to determine direction in 3D space:
- **Row-wise time delay** gives us the **azimuth** direction
- **Column-wise time delay** gives us the **elevation**

Microphone spacing is not a static value and needs to be determined using:

$$\frac{\lambda}{2}$$

This is the Nyquist rule to avoid spatial aliasing. Each microphone uses:

$$x[n] \exp(-2j\pi d k \sin(\theta))$$

Where:
- $d$ is the distance between microphones
- $k$ represents the position of the element in the array
- $\theta$ is the angle of arrival