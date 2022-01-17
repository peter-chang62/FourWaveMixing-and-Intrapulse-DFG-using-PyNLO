function n=LNJundt(lda)
    a1  = 5.35583;
    a2  = 0.100473;
    a3  = 0.20692;
    a4  =  100;
    a5  = 11.34927;
    a6  =  1.5334e-2;
    b1  = 	 4.629e-7;
    b2  = 	 3.862e-8;
    b3  = 	 -0.89e-8;
    b4  = 	 2.657e-5;

    T = 24.5;
    f = (T-24.5)*(T+570.82);    
    
    ne2 = a1 + b1*f + (a2+b2*f)./(lda.^2-(a3+b3*f).^2) + ... 
        (a4 + b4*f)./(lda.^2-a5^2) - a6*lda.^2;
        
    n = sqrt(ne2);