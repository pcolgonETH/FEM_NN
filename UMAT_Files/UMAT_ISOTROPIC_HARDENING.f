!DIR$ FREEFORM
		SUBROUTINE UMAT(STRESS, STATEV, DDSDDE, SSE, SPD, SCD, RPL,&
			 DDSDDT, DRPLDE, DRPLDT, STRAN, DSTRAN, TIME, DTIME, TEMP, DTEMP,&
			 PREDEF, DPRED, CMNAME, NDI, NSHR, NTENS, NSTATV, PROPS, NPROPS,&
			 COORDS, DROT, PNEWDT, CELENT, DFGRD0, DFGRD1, NOEL, NPT, LAYER,&
			 KSPT, KSTEP, KINC)
			 
			 INCLUDE 'ABA_PARAM.INC'
			 
			 CHARACTER*80 CMNAME
			 
			 DIMENSION STRESS(NTENS),STATEV(NSTATV), &
			 DDSDDE(NTENS,NTENS),DDSDDT(NTENS),DRPLDE(NTENS), &
			 STRAN(NTENS),DSTRAN(NTENS),TIME(2),PREDEF(1),DPRED(1), &
			 PROPS(NPROPS),COORDS(3),DROT(3,3),DFGRD0(3,3),DFGRD1(3,3)
			 
			 DIMENSION EELAS(6), EPLAS(6), FLOW(6), HARD(6), OLD_STRESS(NTENS), OLD_EELAS(NTENS), OLD_EPLAS(NTENS), STRESS_INC(NTENS), EELAS_INC(NTENS), EPLAS_INC(NTENS), TOTAL_STRAIN_INC(NTENS), TOTAL_STRAIN(NTENS), STRAIN_RATE(6)
			 
			 !INTEGER, PARAMETER :: NPROPS = 14
			 
			 PARAMETER(ZERO=0.D0, ONE=1.D0, TWO=2.D0, THREE=3.D0, SIX=6.D0, ENUMAX=.4999D0, NEWTON=10, TOLER=1.0D-6)
			 INTEGER STATE_INDEX
			 REAL*8 :: DMAT(6,6)
							
			!Young's Modulus
			EMOD=PROPS(1)
			!Poisson's ratio
			ENU=MIN(PROPS(2), ENUMAX)
			!Lame Parameters
			EBULK3=EMOD/(ONE-TWO*ENU)
			EG2=EMOD/(ONE+ENU)
			EG=EG2/TWO
			EG3=THREE*EG
			ELAM=(EBULK3-EG2)/THREE
			
			DO K1=1, NDI
				DO K2=1, NDI
					DDSDDE(K2, K1)=ELAM
				END DO
				DDSDDE(K1, K1)=EG2+ELAM
			END DO
			
			DO K1=NDI+1, NTENS
				DDSDDE(K1, K1)=EG
			END DO
			
			CALL ROTSIG(STATEV(1:NTENS), DROT, EELAS, 2, NDI, NSHR)
			CALL ROTSIG(STATEV(NTENS+1:2*NTENS), DROT, EPLAS, 2, NDI, NSHR)
			EQPLAS=STATEV(1+2*NTENS)
			
			OLD_EELAS = EELAS
			OLD_EPLAS = EPLAS
			OLD_STRESS = STRESS
			
			!print*,'---------------'
			!print*,'Old Stress'
			!print*,OLD_STRESS
			
			!Elastic Stress calculation
			DO K1=1, NTENS
				DO K2=1, NTENS
				STRESS(K2)=STRESS(K2)+DDSDDE(K2, K1)*DSTRAN(K1)
				END DO
				EELAS(K1)=EELAS(K1)+DSTRAN(K1)
			END DO
			
			!Von Mises stress calculation
			SMISES=(STRESS(1)-STRESS(2))**2+(STRESS(2)-STRESS(3))**2+(STRESS(3)-STRESS(1))**2
			DO K1=NDI+1, NTENS
				SMISES=SMISES+SIX*STRESS(K1)**2
			END DO
			SMISES=SQRT(SMISES/TWO)
			
			!print*,'-------------------'
			!print*,'von Mises Stress'
			!print*,SMISES
			!print*,'-------------------'
			
			NVALUE=NPROPS/2-1
			
			!print*,'NVALUE'
			!print*,NVALUE
			
			!Hardening Subroutine
			CALL UHARD(SYIEL0, HARD, EQPLAS, EQPLASRT, TIME, DTIME, TEMP, &
				DTEMP, NOEL, NPT, LAYER, KSPT, KSTEP, KINC, CMNAME, NSTATV, &
				STATEV, NUMFIELDV, PREDEF, DPRED, NVALUE, PROPS(3),NPROPS)
			
			!Check if current vonMises Stress is greater than yielding stress
			IF (SMISES.GT.(ONE+TOLER)*SYIEL0) THEN
				
				SHYDRO=(STRESS(1)+STRESS(2)+STRESS(3))/THREE
				DO K1=1, NDI
					FLOW(K1)=(STRESS(K1)-SHYDRO)/SMISES
				END DO
				DO K1 = NDI+1, NTENS
					FLOW(K1)=STRESS(K1)/SMISES
				END DO
				
				SYIELD=SYIEL0
				DEQPL=ZERO
				
				DO KEWTON=1, NEWTON
				RHS=SMISES-EG3*DEQPL-SYIELD
				DEQPL=DEQPL+RHS/(EG3+HARD(1))
				CALL UHARD(SYIELD,HARD,EQPLAS+DEQPL,EQPLASRT,TIME,DTIME,TEMP, &
				  DTEMP,NOEL,NPT,LAYER,KSPT,KSTEP,KINC,CMNAME,NSTATV, &
				  STATEV,NUMFIELDV,PREDEF,DPRED,NVALUE,PROPS(3),NPROPS)
				IF(ABS(RHS).LT.TOLER*SYIEL0) CONTINUE
				END DO
				
				!Updated Stress, plastic strain, elastic strain (Main components)
				DO K1=1,NDI
					STRESS(K1)=FLOW(K1)*SYIELD+SHYDRO
					EPLAS(K1)=EPLAS(K1)+THREE/TWO*FLOW(K1)*DEQPL
					EELAS(K1)=EELAS(K1)-THREE/TWO*FLOW(K1)*DEQPL
				END DO
				!Updated Stress, plastic strain, elastic strain (Shear components)
				DO K1=NDI+1,NTENS
					STRESS(K1)=FLOW(K1)*SYIELD
					EPLAS(K1)=EPLAS(K1)+THREE*FLOW(K1)*DEQPL
					EELAS(K1)=EELAS(K1)-THREE*FLOW(K1)*DEQPL
				END DO
				EQPLAS=EQPLAS+DEQPL
				
				!print*,'-------------------'
				!print*,'Yield STRESS'
				!print*,SYIELD
				!print*,SYIEL0
				!print*,'-------------------'
				!print*,'HARD(1)'
				!print*,HARD(1)
				!print*,'-------------------'
				!print*,'Von Mises Stress'
				!print*,SMISES

				
				SPD=DEQPL*(SYIEL0+SYIELD)/TWO
				
				EFFG=EG*SYIELD/SMISES
				EFFG2=TWO*EFFG
				EFFG3=THREE/TWO*EFFG2
				EFFLAM=(EBULK3-EFFG2)/THREE
				EFFHRD=EG3*HARD(1)/(EG3+HARD(1))-EFFG3
				
				!Updated DDSDDE Matrix
				DO K1=1, NDI
					DO K2=1, NDI
						DDSDDE(K2, K1)=EFFLAM
					END DO
					DDSDDE(K1, K1)=EFFG2+EFFLAM
				END DO
				DO K1=NDI+1, NTENS
					DDSDDE(K1, K1)=EFFG
				END DO
				DO K1=1, NTENS
					DO K2=1, NTENS
						DDSDDE(K2, K1)=DDSDDE(K2, K1)+EFFHRD*FLOW(K2)*FLOW(K1)
					END DO
				END DO
			ENDIF
			
			DO K1=1, NTENS
				STATEV(K1)=EELAS(K1)
				STATEV(K1+NTENS)=EPLAS(K1)
			END DO
			
			STATEV(1+2*NTENS)=EQPLAS
			
			STATE_INDEX = 14
			
			DO K1=1, 3
				DO K2=1, 3
					IF (STATE_INDEX .LE. NSTATV) THEN
						STATEV(STATE_INDEX)=DFGRD1(K1,K2)
						STATE_INDEX = STATE_INDEX+1
					END IF
				END DO
			END DO
			
			TOTAL_STRAIN = EELAS + EPLAS
			STATE_INDEX = 23
			
			DO K1=1, NTENS
				IF (STATE_INDEX .LE. NSTATV) THEN 
					STATEV(STATE_INDEX)=TOTAL_STRAIN(K1)
					STATE_INDEX = STATE_INDEX+1
				END IF
			END DO
			
			STATE_INDEX = 29
			
			DO K1=1, NTENS
				DO K2=1, NTENS
					STATEV(STATE_INDEX)=DDSDDE(K1,K2)
					STATE_INDEX = STATE_INDEX+1
				END DO
			END DO
			
			STATE_INDEX = 65
			
			DO K1=1, NTENS
				STATEV(STATE_INDEX) = STRAIN_RATE(K1)
				STATEV(STATE_INDEX+NTENS) = DSTRAN(K1)
				STATEV(STATE_INDEX+2*NTENS) = STRESS(K1)
				STATE_INDEX = STATE_INDEX+1
			END DO
			
			EELAS_INC = EELAS-OLD_EELAS
			EPLAS_INC = EPLAS-OLD_EPLAS
			TOTAL_STRAIN_INC = EELAS_INC+EPLAS_INC
			STRESS_INC = STRESS-OLD_STRESS
			STRAIN_RATE = DSTRAN/DTIME
			
			!do i = 1, 6
				!do j = 1, 6
					!DMAT(i, j) = (STRESS_INC(i)) / (TOTAL_STRAIN_INC(j))
				!end do
			!end do
			
			!DDSDDE = DMAT
			
			!print*,'-------------------'
			!print*, 'Increment'
			!print*, KINC
			!print*,'-------------------'
			!print*,'TOTAL_STRAIN'
			!print*,TOTAL_STRAIN
			!print*,'-------------------'
			!print*,'UMAT STRAN'
			!print*,STRAN+DSTRAN
			!print*,'-------------------'
			!print*,'Stress'
			!print*,STRESS
			!print*,'-------------------'
			!print*,'Stress Increment'
			!print*,STRESS_INC
			!print*,'-------------------'
			!print*,'Strain Increment'
			!print*,DSTRAN
			!print*,'-------------------'
			!print*,'Elastic Strain Increment'
			!print*,EELAS_INC
			!print*,'-------------------'
			!print*,'Plastic Strain Increment'
			!print*,EPLAS_INC
			!print*,'-------------------'
			!print*,'Total Strain Increment'
			!print*,TOTAL_STRAIN_INC
			!print*,'-------------------'
			!print*,'Previous Elastic Strain'
			!print*,OLD_EELAS
			!print*,'-------------------'
			!print*,'Elastic Strain'
			!print*,EELAS
			!print*,'-------------------'
			!print*,'Previous Plastic Strain'
			!print*,OLD_EPLAS
			!print*,'-------------------'
			!print*,'Plastic Strain'
			!print*,EPLAS
			!print*,'-------------------'
			!print*,'Plastic Equivalent Strain'
			!print*,EQPLAS
			!print*,'-------------------'
			!print*,'JACOBIAN'
			!print*,DDSDDE
			!print*,'-------------------'
			!print*,'DTIME'
			!print*,DTIME
			!print*,'-------------------'
			!print*,'Strain Rate'
			!print*,STRAIN_RATE
			!print*,'-------------------------'
			!print*,'DMAT - DERIVED JACOBIAN'
			!print*,DMAT(1,:)
			!print*,DMAT(2,:)
			!print*,DMAT(3,:)
			!print*,DMAT(4,:)
			!print*,DMAT(5,:)
			!print*,DMAT(6,:)
			RETURN
		END SUBROUTINE
			
		SUBROUTINE UHARD(SYIELD,HARD,EQPLAS,EQPLASRT,TIME,DTIME,TEMP, &
				DTEMP,NOEL,NPT,LAYER,KSPT,KSTEP,KINC, &
				CMNAME,NSTATV,STATEV,NUMFIELDV, &
				PREDEF,DPRED,NVALUE,TABLE,NPROPS)

			INCLUDE 'ABA_PARAM.INC'
			CHARACTER*80 CMNAME
			DIMENSION HARD(3),STATEV(NSTATV),TIME(*), &
			  PREDEF(NUMFIELDV),DPRED(*)
			DIMENSION TABLE(2, NVALUE)
			
			PARAMETER(ZERO=0.D0)
			
			!print*,NVALUE
			
			SYIELD = TABLE(1,NVALUE)
			HARD(1) = ZERO
			!SET YIELD STRESS TO LAST VALUE OF TABLE, HARDENING TO ZERO
			
			!NVALUE = (NPROPS - 2) / 2

			!DO K1 = 1, NVALUE
				!TABLE(K1, 1) = PROPS(2*K1+1)  ! Yield stress
				!TABLE(K1, 2) = PROPS(2*K1+2)  ! Plastic strain
			!END DO
			
			 !SYIELD=TABLE(1, 1)
			 !HARD(1)=ZERO
			!IF MORE THAN ONE ENTRY, SEARCH TABLE
			
			IF(NVALUE.GT.1) THEN
				DO K1=1, NVALUE-1
					EQPL1=TABLE(2,K1+1)
					!EQPL0 = TABLE(K1, 2)
					!EQPL1 = TABLE(K1+1, 2)
					IF(EQPLAS.LT.EQPL1) THEN
						EQPL0=TABLE(2, K1)
						!SYIEL0 = TABLE(K1, 1)
						!SYIEL1 = TABLE(K1+1, 1)
						
						!CURRENT YIELD STRESS AND HARDENING
						
						DEQPL=EQPL1-EQPL0
						SYIEL0=TABLE(1, K1)
						SYIEL1=TABLE(1, K1+1)
						DSYIEL=SYIEL1-SYIEL0
						HARD(1)=DSYIEL/DEQPL
						SYIELD=SYIEL0+(EQPLAS-EQPL0)*HARD(1)
						goto 10
					ENDIF
				END DO
		10	    continue
			ENDIF
			
			!print*, NPROPS
			!print*, SYIELD
			!print*, HARD(1)
			
			RETURN
		END SUBROUTINE