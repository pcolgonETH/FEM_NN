		!DIR$ FREEFORM
		SUBROUTINE UMAT(STRESS, STATEV, DDSDDE, SSE, SPD, SCD, RPL,&
			 DDSDDT, DRPLDE, DRPLDT, STRAN, DSTRAN, TIME, DTIME, TEMP, DTEMP,&
			 PREDEF, DPRED, CMNAME, NDI, NSHR, NTENS, NSTATV, PROPS, NPROPS,&
			 COORDS, DROT, PNEWDT, CELENT, DFGRD0, DFGRD1, NOEL, NPT, LAYER,&
			 KSPT, KSTEP, KINC)

			 include 'aba_param.inc'

			 CHARACTER(len=8) :: CMNAME

			 ! assign dimension to subroutine variables
			 DIMENSION STRESS(NTENS),STATEV(NSTATV), &
			 DDSDDE(NTENS,NTENS),DDSDDT(NTENS),DRPLDE(NTENS), &
			 STRAN(NTENS),DSTRAN(NTENS),TIME(2),PREDEF(1),DPRED(1), &
			 PROPS(NPROPS),COORDS(3),DROT(3,3),DFGRD0(3,3),DFGRD1(3,3)
			 INTEGER, PARAMETER :: DP = SELECTED_REAL_KIND(14)
			 REAL*8, dimension(7,6) :: w_L1
			 REAL*8, dimension(7,7) :: w_L2, w_out
			 REAL*8, dimension(7,1) :: b_L1, b_L2, b_out
			 REAL*8, dimension(:,:), allocatable :: x, net_input, output
			 REAL*8 E, NU, LAMBDA, MU
			 REAL*8 :: min_input, scale_input, scaled_data(6,1), data_input_min, data_input_max, data_input_range
		     REAL*8 :: min_output(7,1), scale_output(7,1), scaled_output(7,1), data_output_min(7,1), data_output_max(7,1), data_output_range(7,1)
			 REAL*8 :: sigma_NN(6,1)
			 REAL*8 :: inputs(6,1)
			 REAL*8 :: DMAT(6,6)
			 REAL*8, dimension(6,6) :: DNNDE !Output tangent modulus (derivative of stress with respect to strain)
			 REAL*8 :: DELTA_EPS
			 REAL*8 :: TOTAL_STRAN(6)
			 INTEGER :: i,j
			 CHARACTER*50 FILE_NAME
			 INTEGER FILE_UNIT
			 PARAMETER (FILE_UNIT = 22)
			 CHARACTER*80 INCREMENT_LABEL
			 CHARACTER*30 INC_STR
			 REAL*8 :: JACOBIAN(6,6)
			 REAL*8 :: C11, C12, C44, C55, C66
			 REAL*8 :: TOLERANCE = 1E-12
			 
			  !Input scaling factors (only second element has been scaled)
			  min_input = 1.0264690459130597
			  scale_input = 449.05055234461804
			  data_input_min = -0.004512786
			  data_input_max = -5.8944468e-05
			  data_input_range = 0.0044538415320000005
			  
			  !weights of first hidden layer
			  w_L1(1,:) = (/2.425256849865795102e-19,2.449278384447097778e-01,2.425246510108138189e-19,-1.712005580838497604e-18,-2.199794039759806309e-19,-3.698759615726789627e-17/)
			  w_L1(2,:) = (/-2.433126439418471469e-19,-2.394729405641555786e-01,-2.433120235563877321e-19,1.455306207754610249e-18,2.311863312581846149e-19,3.443655129581965765e-17/)
			  w_L1(3,:) = (/-5.200305116325537554e-19,-2.729429304599761963e-01,-5.200293742592114950e-19,1.787676890455461129e-19,-4.983322458461865831e-20,2.842522795232734260e-17/)
			  w_L1(4,:) = (/2.269433600048821440e-19,2.443564832210540771e-01,2.269427913182110138e-19,-1.648224475539101007e-18,-3.575281402607323868e-19,-3.580733184588538363e-17/)
			  w_L1(5,:) = (/4.360906012149335969e-19,2.628723680973052979e-01,4.360897223355327593e-19,-4.818375664982373697e-19,2.875758023025956417e-21,-2.973612393476667958e-17/)
			  w_L1(6,:) = (/-4.114543708385370663e-20,-2.342308163642883301e-01,-4.114502349354743011e-20,2.603686222040807027e-18,4.824657067758948250e-19,4.047818253695591563e-17/)
			  w_L1(7,:) = (/-4.369258985372473012e-19,4.399313628673553467e-01,-4.369245026699636179e-19,-6.579161740904348449e-18,-1.156692559056895291e-18,-5.543260216028979151e-17/)
			  
			  !biases of first hidden layer
			  b_L1(:,1) = (/3.290401771664619446e-02,1.288499310612678528e-02,-1.252368539571762085e-01,3.634594380855560303e-02,1.014414280652999878e-01,5.535981804132461548e-02,-6.192899942398071289e-01/)
			  
			  !weights of second hidden layer
			  w_L2(1,:) = (/2.686448395252227783e-01,-2.529017329216003418e-01,-3.464812636375427246e-01,2.688899934291839600e-01,3.267240226268768311e-01,-2.014676034450531006e-01,6.478806585073471069e-02/)
			  w_L2(2,:) = (/-2.492012530565261841e-01,2.426908463239669800e-01,2.797564566135406494e-01,-2.465820312500000000e-01,-2.405703812837600708e-01,2.266515940427780151e-01,-3.003183752298355103e-02/)
			  w_L2(3,:) = (/-1.792024970054626465e-01,1.741047352552413940e-01,1.861411035060882568e-01,-1.721789389848709106e-01,-1.528341174125671387e-01,1.518271565437316895e-01,-2.350057065486907959e-01/)
			  w_L2(4,:) = (/-1.203838810324668884e-01,1.205843240022659302e-01,1.083654314279556274e-01,-1.575641483068466187e-01,-1.084394156932830811e-01,1.568087339401245117e-01,-2.526919245719909668e-01/)
			  w_L2(5,:) = (/-1.649326831102371216e-01,1.629761606454849243e-01,1.424934566020965576e-01,-1.589903980493545532e-01,-1.480729877948760986e-01,1.673951447010040283e-01,-3.117335438728332520e-01/)
			  w_L2(6,:) = (/2.100478410720825195e-01,-1.968485563993453979e-01,-2.108698934316635132e-01,2.044711261987686157e-01,2.185615748167037964e-01,-2.107715010643005371e-01,1.208696886897087097e-01/)
			  w_L2(7,:) = (/-1.401671320199966431e-01,1.277176141738891602e-01,7.127039879560470581e-02,-1.416318863630294800e-01,-8.530382066965103149e-02,1.793435811996459961e-01,-5.439798235893249512e-01/)
			  
			  !biases of second hidden layer
			  b_L2(:,1) = (/1.594890505075454712e-01,-8.243059366941452026e-02,-6.447580456733703613e-02,-2.671819329261779785e-01,-1.040974184870719910e-01,-1.227468475699424744e-01,-1.274629980325698853e-01/)
			  
			  !weights of output layer
			  w_out(1,:) = (/4.675057232379913330e-01,-4.523188471794128418e-01,-4.150073528289794922e-01,-3.461524546146392822e-01,-4.169159531593322754e-01,4.347136020660400391e-01,-4.482252001762390137e-01/)
			  w_out(2,:) = (/4.691316485404968262e-01,-4.500093162059783936e-01,-4.147566854953765869e-01,-3.465838730335235596e-01,-4.158429801464080811e-01,4.359523952007293701e-01,-4.484457969665527344e-01/)
			  w_out(3,:) = (/4.704800546169281006e-01,-4.482989013195037842e-01,-4.151451587677001953e-01,-3.456367850303649902e-01,-4.162858426570892334e-01,4.361883997917175293e-01,-4.483885765075683594e-01/)
			  w_out(4,:) = (/5.637722462415695190e-02,2.800009911879897118e-03,-1.107957214117050171e-02,4.248593747615814209e-02,-1.450901105999946594e-02,6.457298994064331055e-02,1.546544879674911499e-01/)
			  w_out(5,:) = (/1.833537369966506958e-01,-1.206024661660194397e-01,8.193699270486831665e-02,3.447428643703460693e-01,4.908771812915802002e-02,4.870349168777465820e-02,1.341567635536193848e-01/)
			  w_out(6,:) = (/6.077102571725845337e-02,-2.524039894342422485e-02,5.354240164160728455e-02,-3.888110071420669556e-02,3.824519366025924683e-02,-1.653717756271362305e-01,1.930812150239944458e-01/)
			  w_out(7,:) = (/-4.631191194057464600e-01,4.446242153644561768e-01,4.165964722633361816e-01,3.420771062374114990e-01,4.203005433082580566e-01,-4.387432932853698730e-01,4.565231800079345703e-01/)
			  
			  !biases of output layer
			  b_out(:,1) = (/-5.608279630541801453e-02,-5.601476877927780151e-02,-5.588286370038986206e-02,2.936488389968872070e-02,-1.550295948982238770e-01,-1.037085950374603271e-01,5.269071832299232483e-02/)
			  
			  !scaling factors for outputs(all 7 outputs (stress vector & deformation) are getting scaled)
			  min_output(:,1) = (/1.0264688327970202, 1.026468680423494, 1.0264688327970202, 0.022676414549411827, -0.1674804861553768, -0.13536605159140502, -1.026528616966745/)
			  scale_output(:,1) = (/4.924608546036892e-09, 2.5369043990307887e-09, 4.924608546036892e-09, 41255873.34083833, 307030246.5627887, 25316635.796826806, 90.01568982800852/)
			  data_output_min(:,1) = (/-411498460.0, -798795840.0, -411498460.0, -2.4788626e-08, -2.7115228e-09, -3.4152798e-08, 0.000294711033347994/)
			  data_output_max(:,1) = (/-5374809.5, -10433456.0, -5374809.5, 2.368932e-08, 3.8024934e-09, 4.484664e-08, 0.0225130598992109/)
			  data_output_range(:,1) = (/406123650.5, 788362384.0, 406123650.5, 4.8477946e-08, 6.5140162e-09, 7.8999438e-08, 0.022218348865862905/)
			  
			  !Strain Update for current increment
			  do i = 1,6
				inputs(i,1) = STRAN(i) + DSTRAN(i)
			  end do
			  
			  !NN inputs = Strain 
			  do i = 1,6
				scaled_data(i,1) = inputs(i,1)
			  end do 
			  
			  !NN input[2](2nd element) is getting scaled accordingly
			  do i = 2,2
				scaled_data(i,1) = -1 + ((inputs(i,1)-data_input_min)/data_input_range)*2
			  end do
			  
			  !Debugging
			  !print*,'-------------------------'
			  !print*,"SCALED DATA: "
			  !print*, scaled_data
			  !print*,'-------------------------'
			  !do i = 1,6
				  !scaled_data(i,1) = (inputs(i,1) - min_input(i,1)) / scale_input(i,1)
				  !scaled_data(i,1) = min_input(i,1) +2*( (inputs(i,1)-data_input_min(i,1))/data_input_range(i,1))
			  !end do
			  
			  print*,scaled_data
			  
			  ! Neural network calculation
			  ! First hidden layer calculation
			  allocate ( x (6, 1))        ! neuron input
			  allocate ( net_input(6, 1)) ! shape same as w_L1.shape[0]
			  allocate ( output(6, 1))    ! shape same as w_L1.shape[0]
			  ! first layer calculation
			  x = scaled_data
			  net_input = matmul(w_L1,x) + b_L1
			  ! Tanh Activation function
			  output = tanh(net_input)

			  ! second hidden layer calculation
			  deallocate ( net_input)
			  deallocate (x)
			  allocate ( x(7, 1))          ! shape same as w_L2.shape[1]
			  allocate ( net_input(7, 1))  ! shape same as w_L2.shape[0]
			  x = output
			  deallocate (output)
			  allocate ( output(7, 1))     ! shape same as w_L2.shape[0]
			  net_input = matmul(w_L2, x) + b_L2
			  !output = max(net_input,ZERO)    ! ReLu function
			  output = tanh(net_input)

			  ! output layer calculation
			  deallocate (net_input)
			  deallocate (x)
			  allocate ( x(7, 1))             ! shape same as w_out.shape[0]
			  allocate ( net_input(7, 1))     ! shape same as w_out.shape[1]
			  x = output
			  deallocate (output)
			  allocate ( output(7, 1))        ! shape same as w_out.shape[1]
			  net_input = matmul(w_out, x) + b_out
			  !output = tanh(net_input)
			  output = net_input
			  
			  !print*,"NN Output Unscaled: "
			  !print*,output
			  !print*,'-------------------------'
			  
			  !Scaling back of the output data
			  do i = 1,7
				  !scaled_output(i,1) = output(i,1) * scale_output(i,1) + min_output(i,1)
				  scaled_output(i,1) = (output(i,1) + 1) * (1/scale_output(i,1)) + data_output_min(i,1)
			  end do
			  
			  do i = 1,6
				  sigma_NN(i,1) = scaled_output(i,1)
			  end do
			  
			  deallocate (output)
			  deallocate (net_input)
			  deallocate (x)
			  
			  !Young's modulus and Poisson's ratio for copper
			  E = PROPS(1)  ! Young's modulus in MPa
			  NU = PROPS(2)  ! Poisson's ratio
			  
			  !Calculate Lame parameters
			  LAMBDA = (E * NU) / ((1.0D0 + NU) * (1.0D0 - 2.0D0 * NU))
			  MU = E / (2.0D0 * (1.0D0 + NU))
			  
			  !for 1st Increment, initialization of DDSDDE(Jacobian/stiffness matrix), NN not suited for inputs of 0
			  if (KINC .LT. 2) then
				  !Initialize DDSDDE to zero
				  DO I = 1, NTENS
					DO J = 1, NTENS
					  DDSDDE(I,J) = 0.0D0
					  !write(*,*) DDSDDE(I,J)
					END DO
				  END DO

				  !Fill in the elasticity tensor
				  DO I = 1, NDI
					DO J = 1, NDI
					  DDSDDE(I,J) = LAMBDA
					END DO
					DDSDDE(I,I) = DDSDDE(I,I) + 2.0D0 * MU
				  END DO

				  !Shear contribution     
				  DO I = NDI+1, NTENS
					DDSDDE(I,I) = MU
				  END DO

				  !Calculate stress
				  DO I = 1, NTENS
					DO J = 1, NTENS
					  STRESS(I) = STRESS(I) + DDSDDE(I,J) * DSTRAN(J)
					  !write(*,*) STRESS(I), DSTRAN(J)
					END DO
				  END DO
				  
				  print*,'-----------------------'
				  print*,'UMAT DDSDDE'
				  print*,DDSDDE
				  
			  end if
			  
			  TOTAL_STRAN = STRAN + DSTRAN
			  
			  !Debugging
			  !write(*,*) NOEL, NPT, KINC, STRAN, DSTRAN, TOTAL_STRAN, STRESS
			  print*,'-------------------------'
			  print*,'-------------------------'
			  print*,'INCREMENT: ',KINC
			  print*,'ELEMENT NUMBER: ',NOEL
			  print*,'INTEGRATION POINT: ',NPT
			  print*,'-------------------------'
			  !print*,'STRAIN'
			  !print*,STRAN
			  print*,'-------------------------'
			  print*,'STRAIN INCREMENT'
			  print*,DSTRAN
			  !print*,'-------------------------'
			  !print*,'TOTAL STRAIN'
			  !print*,TOTAL_STRAN
			  print*,'-------------------------'
			  print*,'ABAQUS STRESS'
			  print*,STRESS
			  
			  !ABAQUS STRESS vector updated with predicted NN stress values
			  STRESS(1) = scaled_output(1,1)
			  STRESS(2) = scaled_output(2,1)
			  STRESS(3) = scaled_output(3,1)
			  STRESS(4) = scaled_output(4,1)
			  STRESS(5) = scaled_output(5,1)
			  STRESS(6) = scaled_output(6,1)
			  
			  IF (KINC .GE. 2) then
				  ! Stiffness matrix calculation
				  ! secant stiffness matrix
				  do i = 1, 6
					do j = 1, 6
					 if (abs(STRAN(j)+DSTRAN(j)) < TOLERANCE) then
					  DMAT(i, j) = 0.0
					 else
					  DMAT(i, j) = (sigma_NN(i,1)) / (STRAN(j)+DSTRAN(j))
					 end if
					end do
				  end do
				  print*,'-------------------------'
				  print*,'PREDICTED STRESS'
				  print*,sigma_NN(:,1)
				  
				  ! in the end, pass Umat and sigma to Abaqus UMAT Dmatrx and sigma_out
				  !DDSDDE   = Dmat
				  !STRESS(:)   = STRESS(:) + sigma_NN_i1(:,1)*50.
				  !print*,'-------------------------'
				  print*,'DMAT - DERIVED JACOBIAN'
				  print*,DMAT(1,:)
				  print*,DMAT(2,:)
				  print*,DMAT(3,:)
				  print*,DMAT(4,:)
				  print*,DMAT(5,:)
				  print*,DMAT(6,:)
				  print*,'--------------------------'
				  !print*,'DDSDDE - ABAQUS JACOBIAN'
				  !print*,DDSDDE
				  
				  C11 = (sigma_NN(2,1)/TOTAL_STRAN(2))
				  C12 = sigma_NN(1,1)/TOTAL_STRAN(2)
				  C44 = sigma_NN(4,1)/TOTAL_STRAN(4)
				  C55 = sigma_NN(5,1)/TOTAL_STRAN(5)
				  C66 = sigma_NN(6,1)/TOTAL_STRAN(6)
				  
				  !print*, C11, C12, C44, C55, C66
				  !print*, sigma_NN(2,1), TOTAL_STRAN(2)
				  
				  JACOBIAN = 0.0
				  JACOBIAN(1,1) = C11
				  JACOBIAN(2,2) = C11
				  JACOBIAN(3,3) = C11
				  JACOBIAN(1,2) = C12
				  JACOBIAN(1,3) = C12
				  JACOBIAN(2,1) = C12
				  JACOBIAN(2,3) = C12
				  JACOBIAN(3,1) = C12
				  JACOBIAN(3,2) = C12
				  JACOBIAN(4,4) = C44
				  JACOBIAN(5,5) = C44
				  JACOBIAN(6,6) = C44
			  
			  
				  !DDSDDE = DMAT
				  DDSDDE = JACOBIAN
				  !DDSDDE(1,1) = JACOBIAN(1,1)
				  
				  print*,'-----------------------'
				  print*,'Hardcoded UMAT DDSDDE'
				  print*,DDSDDE
				  
			  end if
			  
			  RETURN
		END SUBROUTINE