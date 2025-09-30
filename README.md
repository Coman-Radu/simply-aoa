<h1> Multi-source sound source locaizations</h1>

<p> We are using a microphone array with equal spacing between each microphone, along the cardinal diractions, relativaly. Using time delay to determine the angle of arrival along both axiis on the array to determine direction in a 3d space, row-wise time delay gives us the azimuth direction and collumn-wise time delay gives us elevation.<br  /> Microphone spacing is not a static value and needs to be determined using: <br \> $$ \frac{\lamda}{2} $$ This is a Niquist rule to avoid spacial alaising <br  \> each microphone will us: $$ x[n] exp(2j\pi dk \sin(\theta)) $$ where d is the distance between microphone, and k represent the the position of the element in the array</p>

