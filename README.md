# Multi-source Sound Source Localization

We are using a microphone array with equal spacing between each microphone along the cardinal directions. Using time delay to determine the angle of arrival along both axes on the array to determine direction in 3D space:
- **Row-wise time delay** gives us the **azimuth** direction
- **Column-wise time delay** gives us the **elevation**

Microphone spacing is not a static value and needs to be determined using:

$$\frac{\lambda}{2}$$

This is the Nyquist rule to avoid spatial aliasing. 

Each microphone uses the following to determine the **Angle of Arrival**:

$$\Delta t = \lambda \sin(\theta)/2c$$

Where:
- $\Delta t$ is the time delay between mic pairs
- $\lambda/2$ represents the distance between mic pairs in the cardinal direction
- $\theta$ is the angle of arrival 