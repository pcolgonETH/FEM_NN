!DIR$ FREEFORM
		! Define module for neural network parameters
		MODULE NeuralNetParameters
		  IMPLICIT NONE
		  INTEGER, PARAMETER :: hidden_dim = 100, input_dim = 34, kernel_size = 3, output_dim = 42
		  INTEGER, PARAMETER :: num_layers = 3, num_samples = 1000

		  ! Persistent weights and biases
		  REAL(8), ALLOCATABLE :: weights_W_i_L1(:,:), weights_W_f_L1(:,:), weights_W_c_L1(:,:), weights_W_o_L1(:,:)
		  REAL(8), ALLOCATABLE :: biases_W_i_L1(:), biases_W_f_L1(:), biases_W_c_L1(:), biases_W_o_L1(:)
		  REAL(8), ALLOCATABLE :: weights_U_i_L1(:,:), weights_U_f_L1(:,:), weights_U_c_L1(:,:), weights_U_o_L1(:,:)
		  REAL(8), ALLOCATABLE :: weights_output(:,:)
		  REAL(8), ALLOCATABLE :: biases_output(:)
		  
		  ! Hidden states for LSTM
		  REAL(8), ALLOCATABLE :: h_prev(:,:), c_prev(:,:), h_t(:,:), c_t(:,:)

		  ! Scaling parameters for normalization
		  REAL(8), ALLOCATABLE :: feature_min_vals(:), feature_max_vals(:)
		  REAL(8), ALLOCATABLE :: target_min_vals(:), target_max_vals(:)

		  LOGICAL :: LOADED = .FALSE.
		  LOGICAL :: LOADED_SCALING_PARAMS = .FALSE.
		  
		  ! Temporary variable for reading rows
		  REAL(8), DIMENSION(input_dim*hidden_dim) :: temp_row_W_L1_test
		  REAL(8), DIMENSION(hidden_dim*hidden_dim) :: temp_row_U_L1_test, temp_row_U_L2_test, temp_row_U_L3_test, temp_row_W_L2_test, temp_row_W_L3_test
		  REAL(8), DIMENSION(input_dim) :: temp_row_W_L1
		  REAL(8), DIMENSION(hidden_dim) :: temp_row_U_L1, temp_row_L2, temp_row_L3
		  REAL(8), DIMENSION(hidden_dim) :: temp_row_output
		  
		  !Counter variable to track lstm cell calls
		  INTEGER, SAVE :: lstm_cell_call_count = 0

		CONTAINS
		  
		  SUBROUTINE INITIALIZE_PARAMETERS()
			! Allocate arrays for parameters dynamically
			IF (.NOT. LOADED) THEN
			  ALLOCATE(weights_W_i_L1(hidden_dim, input_dim), weights_W_f_L1(hidden_dim, input_dim))
			  ALLOCATE(weights_W_c_L1(hidden_dim, input_dim), weights_W_o_L1(hidden_dim, input_dim))
			  ALLOCATE(biases_W_i_L1(hidden_dim), biases_W_f_L1(hidden_dim))
			  ALLOCATE(weights_U_i_L1(hidden_dim, hidden_dim), weights_U_f_L1(hidden_dim, hidden_dim))
			  ALLOCATE(weights_output(output_dim, hidden_dim), biases_output(output_dim))
			  ALLOCATE(feature_min_vals(input_dim), feature_max_vals(input_dim))
			  ALLOCATE(target_min_vals(output_dim), target_max_vals(output_dim))
			  ALLOCATE(h_prev(num_layers, hidden_dim), c_prev(num_layers, hidden_dim))
			  ALLOCATE(h_t(num_layers, hidden_dim), c_t(num_layers, hidden_dim))
			  LOADED = .TRUE.
			END IF
		  END SUBROUTINE INITIALIZE_PARAMETERS
		  
		  SUBROUTINE LOAD_WEIGHTS()
			IMPLICIT NONE
			INTEGER :: i, j, ios

			IF (.NOT. LOADED) THEN
			  OPEN(UNIT=10, FILE="E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/Layer0_W_i_weight.data", STATUS="OLD")
			  DO i = 1, hidden_dim
				READ(10, *) temp_row_W_L1  ! Read a row into the temporary array
				weights_W_i_L1(i, :) = temp_row_W_L1  ! Assign the row to weights_f_L1
			  END DO
			  CLOSE(10)
			  
			  !OPEN(UNIT=10, FILE="E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/Layer0_W_i_weight.data", STATUS="OLD", ACTION="READ")
			  !if (ios /= 0) then
				!print *, "Error opening file."
				!stop
			  !end if
			  !READ(10, *,iostat=ios) temp_row_W_L1_test
			  !CLOSE(10)
			  
			  ! Assign data to weights_W_i_L1 manually
			  !DO i = 1, hidden_dim
				!DO j = 1, input_dim
				  !weights_W_i_L1(i, j) = temp_row_W_L1_test((i - 1) * input_dim + j)
				!END DO
			  !END DO
			  
			  !PRINT*,weights_W_i_L1(1,:)
			  
			  ! Print the first few elements of weights_f_L1 as a test
			  !PRINT *, "First elements of weights_f_L1:"
			  !PRINT *, weights_W_i_L1(100, 30:34)  ! Adjust the range to display more/less elements as needed
			  
			  OPEN(UNIT=11, FILE="E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/Layer0_U_i_weight.data", STATUS="OLD")
			  DO i = 1, hidden_dim
				READ(11, *) temp_row_U_L1  ! Read a row into the temporary array
				weights_U_i_L1(i, :) = temp_row_U_L1  ! Assign the row to weights_f_L1
			  END DO
			  CLOSE(11)
			  
			  !OPEN(UNIT=11, FILE="E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/Layer0_U_i_weight.data", STATUS="OLD", ACTION="READ")
			  !if (ios /= 0) then
				!print *, "Error opening file."
				!stop
			  !end if
			  !READ(11, *,iostat=ios) temp_row_U_L1_test
			  !CLOSE(11)
			  
			  ! Assign data to weights_W_i_L1 manually
			  !DO i = 1, hidden_dim
				!DO j = 1, hidden_dim
				  !weights_U_i_L1(i, j) = temp_row_U_L1_test((i - 1) * hidden_dim + j)
				!END DO
			  !END DO
			  
			  OPEN(UNIT=12, FILE="E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/Layer0_W_o_weight.data", STATUS="OLD")
			  DO i = 1, hidden_dim
				READ(12, *) temp_row_W_L1  ! Read a row into the temporary array
				weights_W_o_L1(i, :) = temp_row_W_L1  ! Assign the row to weights_f_L1
			  END DO
			  CLOSE(12)
			  
			  !OPEN(UNIT=12, FILE="E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/Layer0_W_o_weight.data", STATUS="OLD", ACTION="READ")
			  !if (ios /= 0) then
				!print *, "Error opening file."
				!stop
			  !end if
			  !READ(12, *,iostat=ios) temp_row_W_L1_test
			  !CLOSE(12)
			  
			  ! Assign data to weights_W_i_L1 manually
			  !DO i = 1, hidden_dim
				!DO j = 1, input_dim
				  !weights_W_o_L1(i, j) = temp_row_W_L1_test((i - 1) * input_dim + j)
				!END DO
			  !END DO
			  
			  OPEN(UNIT=13, FILE="E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/Layer0_U_o_weight.data", STATUS="OLD")
			  DO i = 1, hidden_dim
				READ(13, *) temp_row_U_L1  ! Read a row into the temporary array
				weights_U_o_L1(i, :) = temp_row_U_L1  ! Assign the row to weights_f_L1
			  END DO
			  CLOSE(13)
			  
			  !OPEN(UNIT=13, FILE="E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/Layer0_U_o_weight.data", STATUS="OLD", ACTION="READ")
			  !if (ios /= 0) then
				!print *, "Error opening file."
				!stop
			  !end if
			  !READ(13, *,iostat=ios) temp_row_U_L1_test
			  !CLOSE(13)
			  
			  ! Assign data to weights_W_i_L1 manually
			  !DO i = 1, hidden_dim
				!DO j = 1, hidden_dim
				  !weights_U_o_L1(i, j) = temp_row_U_L1_test((i - 1) * hidden_dim + j)
				!END DO
			  !END DO
			  
			  OPEN(UNIT=14, FILE="E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/Layer0_W_f_weight.data", STATUS="OLD")
			  DO i = 1, hidden_dim
				READ(14, *) temp_row_W_L1  ! Read a row into the temporary array
				weights_W_f_L1(i, :) = temp_row_W_L1  ! Assign the row to weights_f_L1
			  END DO
			  CLOSE(14)
			  
			  !OPEN(UNIT=14, FILE="E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/Layer0_W_f_weight.data", STATUS="OLD", ACTION="READ")
			  !if (ios /= 0) then
				!print *, "Error opening file."
				!stop
			  !end if
			  !READ(14, *,iostat=ios) temp_row_W_L1_test
			  !CLOSE(14)
			  
			  ! Assign data to weights_W_i_L1 manually
			  !DO i = 1, hidden_dim
				!DO j = 1, input_dim
				  !weights_W_f_L1(i, j) = temp_row_W_L1_test((i - 1) * input_dim + j)
				!END DO
			  !END DO
			  
			  OPEN(UNIT=15, FILE="E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/Layer0_U_f_weight.data", STATUS="OLD")
			  DO i = 1, hidden_dim
				READ(15, *) temp_row_U_L1  ! Read a row into the temporary array
				weights_U_f_L1(i, :) = temp_row_U_L1  ! Assign the row to weights_f_L1
			  END DO
			  CLOSE(15)
			  
			  !OPEN(UNIT=15, FILE="E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/Layer0_U_f_weight.data", STATUS="OLD", ACTION="READ")
			  !if (ios /= 0) then
				!print *, "Error opening file."
				!stop
			  !end if
			  !READ(15, *,iostat=ios) temp_row_U_L1_test
			  !CLOSE(15)
			  
			  ! Assign data to weights_W_i_L1 manually
			  !DO i = 1, hidden_dim
				!DO j = 1, hidden_dim
				  !weights_U_f_L1(i, j) = temp_row_U_L1_test((i - 1) * hidden_dim + j)
				!END DO
			  !END DO
			  
			  OPEN(UNIT=16, FILE="E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/Layer0_W_c_weight.data", STATUS="OLD")
			  DO i = 1, hidden_dim
				READ(16, *) temp_row_W_L1  ! Read a row into the temporary array
				weights_W_c_L1(i, :) = temp_row_W_L1  ! Assign the row to weights_f_L1
			  END DO
			  CLOSE(16)
			  
			  !OPEN(UNIT=16, FILE="E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/Layer0_W_c_weight.data", STATUS="OLD", ACTION="READ")
			  !if (ios /= 0) then
				!print *, "Error opening file."
				!stop
			  !end if
			  !READ(16, *,iostat=ios) temp_row_W_L1_test
			  !CLOSE(16)
			  
			  ! Assign data to weights_W_i_L1 manually
			  !DO i = 1, hidden_dim
				!DO j = 1, input_dim
				  !weights_W_c_L1(i, j) = temp_row_W_L1_test((i - 1) * input_dim + j)
				!END DO
			  !END DO
			  
			  OPEN(UNIT=17, FILE="E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/Layer0_U_c_weight.data", STATUS="OLD")
			  DO i = 1, hidden_dim
				READ(17, *) temp_row_U_L1  ! Read a row into the temporary array
				weights_U_c_L1(i, :) = temp_row_U_L1  ! Assign the row to weights_f_L1
			  END DO
			  CLOSE(17)
			  
			  !OPEN(UNIT=17, FILE="E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/Layer0_U_c_weight.data", STATUS="OLD", ACTION="READ")
			  !if (ios /= 0) then
				!print *, "Error opening file."
				!stop
			  !end if
			  !READ(17, *,iostat=ios) temp_row_U_L1_test
			  !CLOSE(17)
			  
			  ! Assign data to weights_W_i_L1 manually
			  !DO i = 1, hidden_dim
				!DO j = 1, hidden_dim
				  !weights_U_c_L1(i, j) = temp_row_U_L1_test((i - 1) * hidden_dim + j)
				!END DO
			  !END DO
			  
			  OPEN(UNIT=18, FILE="E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/Layer1_W_i_weight.data", STATUS="OLD")
			  DO i = 1, hidden_dim
				READ(18, *) temp_row_L2  ! Read a row into the temporary array
				weights_W_i_L2(i, :) = temp_row_L2  ! Assign the row to weights_f_L1
			  END DO
			  CLOSE(18)
			  
			  !OPEN(UNIT=18, FILE="E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/Layer1_W_i_weight.data", STATUS="OLD", ACTION="READ")
			  !if (ios /= 0) then
				!print *, "Error opening file."
				!stop
			  !end if
			  !READ(18, *,iostat=ios) temp_row_W_L2_test
			  !CLOSE(18)
			  
			  ! Assign data to weights_W_i_L1 manually
			  !DO i = 1, hidden_dim
				!DO j = 1, hidden_dim
				  !weights_W_i_L2(i, j) = temp_row_W_L2_test((i - 1) * hidden_dim + j)
				!END DO
			  !END DO
			  
			  OPEN(UNIT=19, FILE="E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/Layer1_U_i_weight.data", STATUS="OLD")
			  DO i = 1, hidden_dim
				READ(19, *) temp_row_L2  ! Read a row into the temporary array
				weights_U_i_L2(i, :) = temp_row_L2  ! Assign the row to weights_f_L1
			  END DO
			  CLOSE(19)
			  
			  !OPEN(UNIT=19, FILE="E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/Layer1_U_i_weight.data", STATUS="OLD", ACTION="READ")
			  !if (ios /= 0) then
				!print *, "Error opening file."
				!stop
			  !end if
			  !READ(19, *,iostat=ios) temp_row_U_L2_test
			  !CLOSE(19)
			  
			  ! Assign data to weights_W_i_L1 manually
			  !DO i = 1, hidden_dim
				!DO j = 1, hidden_dim
				  !weights_U_i_L2(i, j) = temp_row_U_L2_test((i - 1) * hidden_dim + j)
				!END DO
			  !END DO
			  
			  OPEN(UNIT=20, FILE="E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/Layer1_W_o_weight.data", STATUS="OLD")
			  DO i = 1, hidden_dim
				READ(20, *) temp_row_L2  ! Read a row into the temporary array
				weights_W_o_L2(i, :) = temp_row_L2  ! Assign the row to weights_f_L1
			  END DO
			  CLOSE(20)
			  
			  !OPEN(UNIT=20, FILE="E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/Layer1_W_o_weight.data", STATUS="OLD", ACTION="READ")
			  !if (ios /= 0) then
				!print *, "Error opening file."
				!stop
			  !end if
			  !READ(20, *,iostat=ios) temp_row_W_L2_test
			  !CLOSE(20)
			  
			  ! Assign data to weights_W_i_L1 manually
			  !DO i = 1, hidden_dim
				!DO j = 1, hidden_dim
				  !weights_W_o_L2(i, j) = temp_row_W_L2_test((i - 1) * hidden_dim + j)
				!END DO
			  !END DO
			  
			  OPEN(UNIT=21, FILE="E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/Layer1_U_o_weight.data", STATUS="OLD")
			  DO i = 1, hidden_dim
				READ(21, *) temp_row_L2  ! Read a row into the temporary array
				weights_U_o_L2(i, :) = temp_row_L2  ! Assign the row to weights_f_L1
			  END DO
			  CLOSE(21)
			  
			  !OPEN(UNIT=21, FILE="E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/Layer1_U_o_weight.data", STATUS="OLD", ACTION="READ")
			  !if (ios /= 0) then
				!print *, "Error opening file."
				!stop
			  !end if
			  !READ(21, *,iostat=ios) temp_row_U_L2_test
			  !CLOSE(21)
			  
			  ! Assign data to weights_W_i_L1 manually
			  !DO i = 1, hidden_dim
				!DO j = 1, hidden_dim
				  !weights_U_o_L2(i, j) = temp_row_U_L2_test((i - 1) * hidden_dim + j)
				!END DO
			  !END DO
			  
			  OPEN(UNIT=22, FILE="E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/Layer1_W_f_weight.data", STATUS="OLD")
			  DO i = 1, hidden_dim
				READ(22, *) temp_row_L2  ! Read a row into the temporary array
				weights_W_f_L2(i, :) = temp_row_L2  ! Assign the row to weights_f_L1
			  END DO
			  CLOSE(22)
			  
			  !OPEN(UNIT=22, FILE="E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/Layer1_W_f_weight.data", STATUS="OLD", ACTION="READ")
			  !if (ios /= 0) then
				!print *, "Error opening file."
				!stop
			  !end if
			  !READ(22, *,iostat=ios) temp_row_W_L2_test
			  !CLOSE(22)
			  
			  ! Assign data to weights_W_i_L1 manually
			  !DO i = 1, hidden_dim
				!DO j = 1, hidden_dim
				  !weights_W_f_L2(i, j) = temp_row_U_L2_test((i - 1) * hidden_dim + j)
				!END DO
			  !END DO
			  
			  OPEN(UNIT=23, FILE="E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/Layer1_U_f_weight.data", STATUS="OLD")
			  DO i = 1, hidden_dim
				READ(23, *) temp_row_L2  ! Read a row into the temporary array
				weights_U_f_L2(i, :) = temp_row_L2  ! Assign the row to weights_f_L1
			  END DO
			  CLOSE(23)
			  
			  !OPEN(UNIT=23, FILE="E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/Layer1_U_f_weight.data", STATUS="OLD", ACTION="READ")
			  !if (ios /= 0) then
				!print *, "Error opening file."
				!stop
			  !end if
			  !READ(23, *,iostat=ios) temp_row_U_L2_test
			  !CLOSE(23)
			  
			  ! Assign data to weights_W_i_L1 manually
			  !DO i = 1, hidden_dim
				!DO j = 1, hidden_dim
				  !weights_U_f_L2(i, j) = temp_row_U_L2_test((i - 1) * hidden_dim + j)
				!END DO
			  !END DO
			  
			  OPEN(UNIT=24, FILE="E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/Layer1_W_c_weight.data", STATUS="OLD")
			  DO i = 1, hidden_dim
				READ(24, *) temp_row_L2  ! Read a row into the temporary array
				weights_W_c_L2(i, :) = temp_row_L2  ! Assign the row to weights_f_L1
			  END DO
			  CLOSE(24)
			  
			  !OPEN(UNIT=24, FILE="E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/Layer1_W_c_weight.data", STATUS="OLD", ACTION="READ")
			  !if (ios /= 0) then
				!print *, "Error opening file."
				!stop
			  !end if
			  !READ(24, *,iostat=ios) temp_row_W_L2_test
			  !CLOSE(24)
			  
			  ! Assign data to weights_W_i_L1 manually
			  !DO i = 1, hidden_dim
				!DO j = 1, hidden_dim
				  !weights_W_c_L2(i, j) = temp_row_W_L2_test((i - 1) * hidden_dim + j)
				!END DO
			  !END DO
			  
			  OPEN(UNIT=25, FILE="E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/Layer1_U_c_weight.data", STATUS="OLD")
			  DO i = 1, hidden_dim
				READ(25, *) temp_row_L2  ! Read a row into the temporary array
				weights_U_c_L2(i, :) = temp_row_L2  ! Assign the row to weights_f_L1
			  END DO
			  CLOSE(25)
			  
			  !OPEN(UNIT=25, FILE="E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/Layer1_U_c_weight.data", STATUS="OLD", ACTION="READ")
			  !if (ios /= 0) then
				!print *, "Error opening file."
				!stop
			  !end if
			  !READ(25, *,iostat=ios) temp_row_U_L2_test
			  !CLOSE(25)
			  
			  ! Assign data to weights_W_i_L1 manually
			  !DO i = 1, hidden_dim
				!DO j = 1, hidden_dim
				  !weights_U_c_L2(i, j) = temp_row_U_L2_test((i - 1) * hidden_dim + j)
				!END DO
			  !END DO
			  
			  OPEN(UNIT=26, FILE="E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/Layer2_W_i_weight.data", STATUS="OLD")
			  DO i = 1, hidden_dim
				READ(26, *) temp_row_L3  ! Read a row into the temporary array
				weights_W_i_L3(i, :) = temp_row_L3  ! Assign the row to weights_f_L1
			  END DO
			  CLOSE(26)
			  
			  !OPEN(UNIT=26, FILE="E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/Layer2_W_i_weight.data", STATUS="OLD", ACTION="READ")
			  !if (ios /= 0) then
				!print *, "Error opening file."
				!stop
			  !end if
			  !READ(26, *,iostat=ios) temp_row_W_L3_test
			  !CLOSE(26)
			  
			  ! Assign data to weights_W_i_L1 manually
			  !DO i = 1, hidden_dim
				!DO j = 1, hidden_dim
				  !weights_W_i_L3(i, j) = temp_row_W_L3_test((i - 1) * hidden_dim + j)
				!END DO
			  !END DO
			  
			  OPEN(UNIT=27, FILE="E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/Layer2_U_i_weight.data", STATUS="OLD")
			  DO i = 1, hidden_dim
				READ(27, *) temp_row_L3  ! Read a row into the temporary array
				weights_U_i_L3(i, :) = temp_row_L3  ! Assign the row to weights_f_L1
			  END DO
			  CLOSE(27)
			  
			  !OPEN(UNIT=27, FILE="E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/Layer2_U_i_weight.data", STATUS="OLD", ACTION="READ")
			  !if (ios /= 0) then
				!print *, "Error opening file."
				!stop
			  !end if
			  !READ(27, *,iostat=ios) temp_row_U_L3_test
			  !CLOSE(27)
			  
			  ! Assign data to weights_W_i_L1 manually
			  !DO i = 1, hidden_dim
				!DO j = 1, hidden_dim
				  !weights_U_i_L3(i, j) = temp_row_U_L3_test((i - 1) * hidden_dim + j)
				!END DO
			  !END DO
			  
			  OPEN(UNIT=28, FILE="E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/Layer2_W_o_weight.data", STATUS="OLD")
			  DO i = 1, hidden_dim
				READ(28, *) temp_row_L3  ! Read a row into the temporary array
				weights_W_o_L3(i, :) = temp_row_L3  ! Assign the row to weights_f_L1
			  END DO
			  CLOSE(28)
			  
			  !OPEN(UNIT=28, FILE="E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/Layer2_W_o_weight.data", STATUS="OLD", ACTION="READ")
			  !if (ios /= 0) then
				!print *, "Error opening file."
				!stop
			  !end if
			  !READ(28, *,iostat=ios) temp_row_W_L3_test
			  !CLOSE(28)
			  
			  ! Assign data to weights_W_i_L1 manually
			  !DO i = 1, hidden_dim
				!DO j = 1, hidden_dim
				  !weights_W_o_L3(i, j) = temp_row_W_L3_test((i - 1) * hidden_dim + j)
				!END DO
			  !END DO
			  
			  OPEN(UNIT=29, FILE="E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/Layer2_U_o_weight.data", STATUS="OLD")
			  DO i = 1, hidden_dim
				READ(29, *) temp_row_L3  ! Read a row into the temporary array
				weights_U_o_L3(i, :) = temp_row_L3  ! Assign the row to weights_f_L1
			  END DO
			  CLOSE(29)
			  
			  !OPEN(UNIT=29, FILE="E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/Layer2_U_o_weight.data", STATUS="OLD", ACTION="READ")
			  !if (ios /= 0) then
				!print *, "Error opening file."
				!stop
			  !end if
			  !READ(29, *,iostat=ios) temp_row_U_L3_test
			  !CLOSE(29)
			  
			  ! Assign data to weights_W_i_L1 manually
			  !DO i = 1, hidden_dim
				!DO j = 1, hidden_dim
				  !weights_U_o_L3(i, j) = temp_row_U_L3_test((i - 1) * hidden_dim + j)
				!END DO
			  !END DO
			  
			  OPEN(UNIT=30, FILE="E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/Layer2_W_f_weight.data", STATUS="OLD")
			  DO i = 1, hidden_dim
				READ(30, *) temp_row_L3  ! Read a row into the temporary array
				weights_W_f_L3(i, :) = temp_row_L3  ! Assign the row to weights_f_L1
			  END DO
			  CLOSE(30)
			  
			  !OPEN(UNIT=30, FILE="E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/Layer2_W_f_weight.data", STATUS="OLD", ACTION="READ")
			  !if (ios /= 0) then
				!print *, "Error opening file."
				!stop
			  !end if
			  !READ(30, *,iostat=ios) temp_row_W_L3_test
			  !CLOSE(30)
			  
			  ! Assign data to weights_W_i_L1 manually
			  !DO i = 1, hidden_dim
				!DO j = 1, hidden_dim
				  !weights_W_f_L3(i, j) = temp_row_W_L3_test((i - 1) * hidden_dim + j)
				!END DO
			  !END DO
			  
			  OPEN(UNIT=31, FILE="E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/Layer2_U_f_weight.data", STATUS="OLD")
			  DO i = 1, hidden_dim
				READ(31, *) temp_row_L3  ! Read a row into the temporary array
				weights_U_f_L3(i, :) = temp_row_L3  ! Assign the row to weights_f_L1
			  END DO
			  CLOSE(31)
			  
			  !OPEN(UNIT=31, FILE="E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/Layer2_U_f_weight.data", STATUS="OLD", ACTION="READ")
			  !if (ios /= 0) then
				!print *, "Error opening file."
				!stop
			  !end if
			  !READ(31, *,iostat=ios) temp_row_U_L3_test
			  !CLOSE(31)
			  
			  ! Assign data to weights_W_i_L1 manually
			  !DO i = 1, hidden_dim
				!DO j = 1, hidden_dim
				  !weights_U_f_L3(i, j) = temp_row_U_L3_test((i - 1) * hidden_dim + j)
				!END DO
			  !END DO
			  
			  OPEN(UNIT=32, FILE="E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/Layer2_W_c_weight.data", STATUS="OLD")
			  DO i = 1, hidden_dim
				READ(32, *) temp_row_L3  ! Read a row into the temporary array
				weights_W_c_L3(i, :) = temp_row_L3  ! Assign the row to weights_f_L1
			  END DO
			  CLOSE(32)
			  
			  !OPEN(UNIT=32, FILE="E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/Layer2_W_c_weight.data", STATUS="OLD", ACTION="READ")
			  !if (ios /= 0) then
				!print *, "Error opening file."
				!stop
			  !end if
			  !READ(32, *,iostat=ios) temp_row_W_L3_test
			  !CLOSE(32)
			  
			  ! Assign data to weights_W_i_L1 manually
			  !DO i = 1, hidden_dim
				!DO j = 1, hidden_dim
				  !weights_W_c_L3(i, j) = temp_row_W_L3_test((i - 1) * hidden_dim + j)
				!END DO
			  !END DO
			  
			  OPEN(UNIT=33, FILE="E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/Layer2_U_c_weight.data", STATUS="OLD")
			  DO i = 1, hidden_dim
				READ(33, *) temp_row_L3  ! Read a row into the temporary array
				weights_U_c_L3(i, :) = temp_row_L3  ! Assign the row to weights_f_L1
			  END DO
			  CLOSE(33)
			  
			  !OPEN(UNIT=33, FILE="E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/Layer2_U_c_weight.data", STATUS="OLD", ACTION="READ")
			  !if (ios /= 0) then
				!print *, "Error opening file."
				!stop
			  !end if
			  !READ(33, *,iostat=ios) temp_row_U_L3_test
			  !CLOSE(33)
			  
			  ! Assign data to weights_W_i_L1 manually
			  !DO i = 1, hidden_dim
				!DO j = 1, hidden_dim
				  !weights_U_c_L3(i, j) = temp_row_U_L3_test((i - 1) * hidden_dim + j)
				!END DO
			  !END DO
			  
			  OPEN(UNIT=34, FILE='E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/Layer0_W_i_bias.data', STATUS='OLD', ACTION='READ')
			  READ(34, *) biases_W_i_L1
			  CLOSE(34)
			  
			  OPEN(UNIT=35, FILE='E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/Layer0_U_i_bias.data', STATUS='OLD', ACTION='READ')
			  READ(35, *) biases_U_i_L1
			  CLOSE(35)
			  
			  OPEN(UNIT=36, FILE='E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/Layer0_W_o_bias.data', STATUS='OLD', ACTION='READ')
			  READ(36, *) biases_W_o_L1
			  CLOSE(36)
			  
			  OPEN(UNIT=37, FILE='E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/Layer0_U_o_bias.data', STATUS='OLD', ACTION='READ')
			  READ(37, *) biases_U_o_L1
			  CLOSE(37)
			  
			  OPEN(UNIT=38, FILE='E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/Layer0_W_f_bias.data', STATUS='OLD', ACTION='READ')
			  READ(38, *) biases_W_f_L2
			  CLOSE(38)
			  
			  OPEN(UNIT=39, FILE='E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/Layer0_U_f_bias.data', STATUS='OLD', ACTION='READ')
			  READ(39, *) biases_U_f_L1
			  CLOSE(39)
			  
			  OPEN(UNIT=40, FILE='E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/Layer1_W_i_bias.data', STATUS='OLD', ACTION='READ')
			  READ(40, *) biases_W_i_L2
			  CLOSE(40)
			  
			  OPEN(UNIT=41, FILE='E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/Layer1_U_i_bias.data', STATUS='OLD', ACTION='READ')
			  READ(41, *) biases_U_i_L2
			  CLOSE(41)
			  
			  OPEN(UNIT=42, FILE='E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/Layer1_W_o_bias.data', STATUS='OLD', ACTION='READ')
			  READ(42, *) biases_W_o_L2
			  CLOSE(42)
			  
			  OPEN(UNIT=43, FILE='E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/Layer1_U_o_bias.data', STATUS='OLD', ACTION='READ')
			  READ(43, *) biases_U_o_L2
			  CLOSE(43)
			  
			  OPEN(UNIT=44, FILE='E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/Layer1_W_f_bias.data', STATUS='OLD', ACTION='READ')
			  READ(44, *) biases_W_f_L2
			  CLOSE(44)
			  
			  OPEN(UNIT=45, FILE='E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/Layer1_U_f_bias.data', STATUS='OLD', ACTION='READ')
			  READ(45, *) biases_U_f_L2
			  CLOSE(45)
			  
			  OPEN(UNIT=46, FILE='E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/Layer1_W_c_bias.data', STATUS='OLD', ACTION='READ')
			  READ(46, *) biases_W_c_L2
			  CLOSE(46)
			  
			  OPEN(UNIT=47, FILE='E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/Layer1_U_c_bias.data', STATUS='OLD', ACTION='READ')
			  READ(47, *) biases_U_c_L2
			  CLOSE(47)
			  
			  OPEN(UNIT=48, FILE='E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/Layer2_W_i_bias.data', STATUS='OLD', ACTION='READ')
			  READ(48, *) biases_W_i_L3
			  CLOSE(48)
			  
			  OPEN(UNIT=49, FILE='E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/Layer2_U_i_bias.data', STATUS='OLD', ACTION='READ')
			  READ(49, *) biases_U_i_L3
			  CLOSE(49)
			  
			  OPEN(UNIT=50, FILE='E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/Layer2_W_o_bias.data', STATUS='OLD', ACTION='READ')
			  READ(50, *) biases_W_o_L3
			  CLOSE(50)
			  
			  OPEN(UNIT=51, FILE='E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/Layer2_U_o_bias.data', STATUS='OLD', ACTION='READ')
			  READ(51, *) biases_U_o_L3
			  CLOSE(51)
			  
			  OPEN(UNIT=52, FILE='E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/Layer2_W_f_bias.data', STATUS='OLD', ACTION='READ')
			  READ(52, *) biases_W_f_L3
			  CLOSE(52)
			  
			  OPEN(UNIT=53, FILE='E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/Layer2_U_f_bias.data', STATUS='OLD', ACTION='READ')
			  READ(53, *) biases_U_f_L3
			  CLOSE(53)
			  
			  OPEN(UNIT=54, FILE='E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/Layer2_W_c_bias.data', STATUS='OLD', ACTION='READ')
			  READ(54, *) biases_W_c_L3
			  CLOSE(54)
			  
			  OPEN(UNIT=55, FILE='E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/Layer2_U_c_bias.data', STATUS='OLD', ACTION='READ')
			  READ(55, *) biases_U_c_L3
			  CLOSE(55)
			  
			  OPEN(UNIT=56, FILE="E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/OutputLayer_weight.data", STATUS="OLD")
			  DO i = 1, output_dim
				READ(56, *) temp_row_output  ! Read a row into the temporary array
				weights_output(i, :) = temp_row_output  ! Assign the row to weights_f_L1
			  END DO
			  CLOSE(56)
			  
			  OPEN(UNIT=57, FILE='E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/OutputLayer_bias.data', STATUS='OLD', ACTION='READ')
			  READ(57, *) biases_output
			  CLOSE(57)
			  
			  LOADED = .TRUE.
			END IF
		  END SUBROUTINE LOAD_WEIGHTS
		  
		  SUBROUTINE LOAD_SCALING_PARAMS()
			IMPLICIT NONE
			INTEGER :: i, ios, num_values
			character(len=2000) :: temp_line
			REAL(8), DIMENSION(input_dim) :: temp_feature_min_vals, temp_feature_max_vals
			REAL(8), DIMENSION(output_dim) :: temp_target_min_vals, temp_target_max_vals

			IF (.NOT. LOADED_SCALING_PARAMS) THEN
				! Load feature scaling parameters (min and max values)
				OPEN(UNIT=50, FILE="E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/feature_min_vals.txt", STATUS="OLD", ACTION="READ")
				if (ios /= 0) then
					print *, "Error opening file."
					stop
				end if
				READ(50, *,iostat=ios) feature_min_vals
				CLOSE(50)
				
				!Print*,SIZE(temp_feature_min_vals)
				!Print*,temp_feature_min_vals
				
				OPEN(UNIT=51, FILE="E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/feature_max_vals.txt", STATUS="OLD", ACTION="READ")
				if (ios /= 0) then
					print *, "Error opening file."
					stop
				end if
				READ(51, *,iostat=ios) feature_max_vals
				CLOSE(51)

				! Load target scaling parameters similarly
				OPEN(UNIT=52, FILE="E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/target_min_vals.txt", STATUS="OLD", ACTION="READ")
				if (ios /= 0) then
					print *, "Error opening file."
					stop
				end if
				READ(52, *,iostat=ios) target_min_vals
				CLOSE(52)

				OPEN(UNIT=53, FILE="E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Uniaxial_02/target_max_vals.txt", STATUS="OLD", ACTION="READ")
				if (ios /= 0) then
					print *, "Error opening file."
					stop
				end if
				READ(53, *,iostat=ios) target_max_vals
				CLOSE(53)

				LOADED_SCALING_PARAMS = .TRUE.
			END IF
		  END SUBROUTINE LOAD_SCALING_PARAMS
		  
		  SUBROUTINE SCALE_FEATURES(data, scaled_data)
			REAL(8), DIMENSION(:), INTENT(IN) :: data       ! 1D array of input data, shape (num_features)
			REAL(8), DIMENSION(:), INTENT(OUT) :: scaled_data  ! Scaled output, same shape as data
			INTEGER :: i, num_features
			REAL(8) :: min_val, max_val
			
			num_features = SIZE(data)
			
			!PRINT*, "feature_min_vals:"
			!PRINT*, feature_min_vals
			!PRINT*, "feature_max_vals:"
			!PRINT*, feature_max_vals
			
			! Loop over each feature to apply scaling
			DO i = 1, num_features
				min_val = feature_min_vals(i)
				max_val = feature_max_vals(i)

				! Avoid division by zero if min and max values are the same
				IF (max_val /= min_val) THEN
					scaled_data(i) = (data(i) - min_val) / (max_val - min_val)
				ELSE
					scaled_data(i) = 0.0D0
				END IF
			END DO	
		  END SUBROUTINE SCALE_FEATURES

		  SUBROUTINE SCALE_BACK_OUTPUT(scaled_data, original_data)
			REAL(8), DIMENSION(:), INTENT(IN) :: scaled_data  ! Scaled output data, shape (num_outputs)
			REAL(8), DIMENSION(:), INTENT(OUT) :: original_data  ! Scaled-back output data, same shape as scaled_data
			INTEGER :: i, num_outputs
			REAL(8) :: min_val, max_val

			num_outputs = SIZE(scaled_data)  ! Determine number of outputs dynamically

			! Loop over each output to scale back
			DO i = 1, num_outputs
				min_val = target_min_vals(i)
				max_val = target_max_vals(i)

				! Apply the inverse scaling
				original_data(i) = scaled_data(i) * (max_val - min_val) + min_val
			END DO
		  END SUBROUTINE SCALE_BACK_OUTPUT
		  
		END MODULE NeuralNetParameters
		
		SUBROUTINE UMAT(STRESS, STATEV, DDSDDE, SSE, SPD, SCD, RPL,&
			 DDSDDT, DRPLDE, DRPLDT, STRAN, DSTRAN, TIME, DTIME, TEMP, DTEMP,&
			 PREDEF, DPRED, CMNAME, NDI, NSHR, NTENS, NSTATV, PROPS, NPROPS,&
			 COORDS, DROT, PNEWDT, CELENT, DFGRD0, DFGRD1, NOEL, NPT, LAYER,&
			 KSPT, KSTEP, KINC)
			 
			 USE NeuralNetParameters
			 
			 INCLUDE 'ABA_PARAM.INC'
			 
			 CHARACTER*80 CMNAME
			 
			 DIMENSION STRESS(NTENS),STATEV(NSTATV), &
			 DDSDDE(NTENS,NTENS),DDSDDT(NTENS),DRPLDE(NTENS), &
			 STRAN(NTENS),DSTRAN(NTENS),TIME(2),PREDEF(1),DPRED(1), &
			 PROPS(NPROPS),COORDS(3),DROT(3,3),DFGRD0(3,3),DFGRD1(3,3)
			 
			 DIMENSION EELAS(6), EPLAS(6), FLOW(6), HARD(6), OLD_STRESS(NTENS), OLD_EELAS(NTENS), OLD_EPLAS(NTENS), STRESS_INC(NTENS), EELAS_INC(NTENS), EPLAS_INC(NTENS), TOTAL_STRAIN_INC(NTENS), TOTAL_STRAIN(NTENS), STRAIN_RATE(6)
			 
			 !INTEGER, PARAMETER :: NPROPS = 14
			 
			 PARAMETER(ZERO=0.D0, ONE=1.D0, TWO=2.D0, THREE=3.D0, SIX=6.D0, ENUMAX=.4999D0, NEWTON=10, TOLER=1.0D-6)
			 INTEGER STATE_INDEX, K1, K2, K3
			 REAL*8 :: DMAT(6,6)
			 REAL*8, dimension(6, 1) :: sigma_NN
			 REAL*8, dimension(36, 1) :: ddsdde_NN
			 
			 ! Declare the input tensor, output tensor, hidden states, etc.
			 real(8), dimension(hidden_dim) :: h, c, h_new, c_new
			 real(8), dimension(input_dim) :: input_tensor, scaled_input
			 real(8), dimension(output_dim) :: output, scaled_output
			 
			 ! Declarations
			 REAL(8) :: h_prev(num_layers, hidden_dim)
			 REAL(8) :: c_prev(num_layers, hidden_dim)
			 REAL(8) :: h_t(num_layers, hidden_dim)
			 REAL(8) :: c_t(num_layers, hidden_dim)
			 
			 ! Initialize parameters if not already initialized
			 IF (.NOT. LOADED) CALL INITIALIZE_PARAMETERS()
			 
			 ! Ensure weights are loaded only once
			 IF (.NOT. LOADED) CALL LOAD_WEIGHTS()
			 
			 CALL LOAD_SCALING_PARAMS()
			 
			 ! Load hidden states (h) and cell states (c) from STATEV
			 h = STATEV(21:20 + hidden_dim)
			 c = STATEV(71:70 + hidden_dim)
			
			 ! Prepare input_tensor with data for current time step
			 input_tensor(1:6) = EELAS
			 input_tensor(7:12) = EPLAS
			 ! (add additional input features as needed)
			
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
						STATEV(STATE_INDEX)=DFGRD0(K1,K2)
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
			!print*,'-------------------------'
			!print*,'DMAT - DERIVED JACOBIAN'
			!print*,DMAT(1,:)
			!print*,DMAT(2,:)
			!print*,DMAT(3,:)
			!print*,DMAT(4,:)
			!print*,DMAT(5,:)
			!print*,DMAT(6,:)
			!print*,'-------------------------'
			!print*,'STRAIN_RATE'
			!print*,STRAIN_RATE
			!print*,'-------------------------'
			!print*,'DFGRD0'
			!print*,DFGRD0
			
			if (KINC == 1) then
				h = 0.0
				c = 0.0
			end if
			
			! Prepare input tensor for cuurent time step
			input_tensor(1:6) = EELAS !!SOMETHING WRONG
			input_tensor(7:12) = EPLAS !!SOMETHING WRONG
			input_tensor(13:18) = DSTRAN !!SOMETHING WRONG (Factor 2 missing) -> always 2 increment per increment -> perhaps reason
			input_tensor(19) = EQPLAS
			input_tensor(20:25) = STRAIN_RATE
			input_tensor(26:34) = reshape(DFGRD0, (/9/))
			
			!PRINT*,"Unscaled input"
			!PRINT*,input_tensor
			
			CALL SCALE_FEATURES(input_tensor, scaled_input)
			
			Print*,'Increment number', KINC, 'Element', NOEL, 'Integration point', NPT
			
			!PRINT *, "Input tensor:"
			!PRINT *, input_tensor  
			
			!PRINT*, "Scaled input tensor:"
			!PRINT*, scaled_input
			
			!PRINT *, "Output:"
			!PRINT *, output 
			
			! Retrieve previous hidden and cell states
		    h_prev = RESHAPE(STATEV(65 : num_layers * hidden_dim + 64), (/num_layers, hidden_dim/))
		    c_prev = RESHAPE(STATEV(num_layers * hidden_dim + 65: 2 * num_layers * hidden_dim + 64), (/num_layers, hidden_dim/))
			!PRINT*,"h_prev:"
			!PRINT*,h_prev
			!PRINT*,"c_prev:"
			!PRINT*,c_prev
			!PRINT*,"W_i_Layer0:"
			!PRINT*,weights_W_i_L1
			
			CALL lstm_forward(scaled_input, h_prev, c_prev, h_t, c_t, output)
			
			STATEV(65 : 64 + num_layers * hidden_dim) = RESHAPE(h_t, (/num_layers * hidden_dim/))
			STATEV(65 + num_layers * hidden_dim : 64 + 2 * num_layers * hidden_dim) = RESHAPE(c_t, (/num_layers * hidden_dim/))
			
			!PRINT*,"h_t:"
			!PRINT*,h_t
			!PRINT*,"c_t:"
			!PRINT*,c_t
			
			!PRINT*, "Output tensor:"
			!PRINT*, output
			
			CALL SCALE_BACK_OUTPUT(output, scaled_output)
			
			PRINT*, "Scaled output tensor:"
			PRINT*, scaled_output
			
			!if (KINC > 10) then
				
				!do K1 = 1, 6
					!STRESS(K1) = sigma_NN(K1, 1)
				!end do
				
				!K3 = 0
				
				!do K1 = 1, 6
					!do K2 = 1, 6
						!K3 = K3 + 1
						!DDSDDE(K1, K2) = ddsdde_NN(K3, 1)
					!end do 
				!end do
				
			!end if	
			
			RETURN
		END SUBROUTINE
		
		SUBROUTINE sigmoid(x, y)
		  USE NeuralNetParameters
		  IMPLICIT NONE
		  REAL(8), INTENT(IN) :: x(hidden_dim)
		  REAL(8), INTENT(OUT) :: y(hidden_dim)
		  INTEGER :: i, n

		  n = SIZE(x)
		  !Print*,'Sigmoid', n
		  DO i = 1, n
			y(i) = 1.0D0 / (1.0D0 + EXP(-x(i)))
		  END DO
		  !Print*,SIZE(y)
		END SUBROUTINE sigmoid

		SUBROUTINE tanh_activation(x, y)
		  USE NeuralNetParameters
		  IMPLICIT NONE
		  REAL(8), INTENT(IN) :: x(hidden_dim)
		  REAL(8), INTENT(OUT) :: y(hidden_dim)
		  INTEGER :: i, n

		  n = SIZE(x)
		  !Print*,'TANH', n
		  DO i = 1, n
			y(i) = TANH(x(i))
		  END DO
		  !Print*,SIZE(y)
		END SUBROUTINE tanh_activation
		
		SUBROUTINE lstm_cell(x_t, h_prev, c_prev, h_t, c_t, &
							 W_i, U_i, b_W_i, b_U_i, &
							 W_f, U_f, b_W_f, b_U_f, &
							 W_c, U_c, b_W_c, b_U_c, &
							 W_o, U_o, b_W_o, b_U_o, &
							 layer)
		  USE NeuralNetParameters
		  IMPLICIT NONE
		  INTEGER, INTENT(IN) :: layer
		  REAL(8), INTENT(IN) :: x_t(:)
		  REAL(8), INTENT(IN) :: h_prev(hidden_dim)
		  REAL(8), INTENT(IN) :: c_prev(hidden_dim)
		  REAL(8), INTENT(IN) :: W_i(:, :), U_i(:, :), b_W_i(hidden_dim), b_U_i(hidden_dim)
		  REAL(8), INTENT(IN) :: W_f(:, :), U_f(:, :), b_W_f(hidden_dim), b_U_f(hidden_dim)
		  REAL(8), INTENT(IN) :: W_c(:, :), U_c(:, :), b_W_c(hidden_dim), b_U_c(hidden_dim)
		  REAL(8), INTENT(IN) :: W_o(:, :), U_o(:, :), b_W_o(hidden_dim), b_U_o(hidden_dim)
		  REAL(8), INTENT(OUT) :: h_t(hidden_dim)
		  REAL(8), INTENT(OUT) :: c_t(hidden_dim)

		  ! Local variables
		  REAL(8) :: i_t(hidden_dim), f_t(hidden_dim), c_hat_t(hidden_dim), o_t(hidden_dim)
		  REAL(8) :: temp_vec(hidden_dim)
		  
		  !Increment the Counter
		  lstm_cell_call_count = lstm_cell_call_count + 1
		  !Print*,'lstm_cell_call_count = ', lstm_cell_call_count
		  
		  ! Input gate
		  temp_vec = MATMUL(W_i, x_t) + b_W_i + MATMUL(U_i, h_prev) + b_U_i
		  !PRINT*,'Input gate size temp_vec'
		  !Print*,SIZE(temp_vec)
		  !if (layer==1) then
		  PRINT*,"Layer:"
		  PRINT*,layer
			!PRINT*,"Input gate before sigmoid:"
			!PRINT*,temp_vec
			!Print*,"W_i:"
		    !Print*,W_i
			PRINT*,"x_t:"
			PRINT*,x_t
			!PRINT*,"W_i_first_row:"
			!PRINT*,W_i(1,:)
			!PRINT*,"MATMUL(W_i,x_t):"
			!PRINT*,MATMUL(W_i, x_t)
			PRINT*,"W_i(x):"
			PRINT*,MATMUL(W_i, x_t) + b_W_i
			!Print*,"b_W_i:"
		    !Print*,b_W_i
			!Print*,"U_i:"
		    !Print*,U_i
			!PRINT*,"MATMUL(U_i, h_prev):"
			!PRINT*,MATMUL(U_i, h_prev)
			!Print*,"b_U_i:"
		    !Print*,b_U_i
			!Print*,"h_prev:"
		    !Print*,h_prev
		  !end if
		  CALL sigmoid(temp_vec, i_t)
		  !PRINT*,'Input gate size i_t'
		  !Print*,SIZE(i_t)
		  if (layer==1) then
			!PRINT*,"Input gate outputs:"
			!PRINT*,i_t
		  end if
		  
		  !Print*,"W_f:"
		  !Print*,W_f
		  !Print*,"x_t:"
		  !Print*,x_t
		  !Print*,"b_W_f:"
		  !Print*,b_W_f
		  !Print*,"U_f:"
		  !Print*,U_f
		  !Print*,"h_prev:"
		  !Print*,h_prev
		  !Print*,"b_U_f:"
		  !Print*,b_U_f
		  
		  ! Forget gate
		  temp_vec = MATMUL(W_f, x_t) + b_W_f + MATMUL(U_f, h_prev) + b_U_f
		  !PRINT*,'Forget gate size temp_vec'
		  !Print*,SIZE(temp_vec)
		  CALL sigmoid(temp_vec, f_t)
		  !PRINT*,'Input gate size f_t'
		  !Print*,SIZE(f_t)

		  ! Cell candidate
		  temp_vec = MATMUL(W_c, x_t) + b_W_c + MATMUL(U_c, h_prev) + b_U_c
		  !PRINT*,'Cell candidate size temp_vec'
		  !Print*,SIZE(temp_vec)
		  CALL tanh_activation(temp_vec, c_hat_t)
		  !PRINT*,'Input gate size c_hat_t'
		  !Print*,SIZE(c_hat_t)

		  ! Update cell state
		  c_t = f_t * c_prev + i_t * c_hat_t

		  ! Output gate
		  temp_vec = MATMUL(W_o, x_t) + b_W_o + MATMUL(U_o, h_prev) + b_U_o
		  !PRINT*,'Output gate size temp_vec'
		  !Print*,SIZE(temp_vec)
		  CALL sigmoid(temp_vec, o_t)
		  !PRINT*,'Input gate size o_t'
		  !Print*,SIZE(o_t)

		  ! Update hidden state
		  CALL tanh_activation(c_t, temp_vec)
		  !PRINT*,'hidden state size temp_vec'
		  !Print*,SIZE(temp_vec)
		  h_t = o_t * temp_vec
		  
		  !Print*,'LSTM CELL h_t'
		  !Print*,'Layer', layer
		  !Print*, hidden_dim
		  !Print*,h_t
		  	  
		END SUBROUTINE lstm_cell
		
		SUBROUTINE lstm_forward(x, h_prev, c_prev, h_t, c_t, output)
		  USE NeuralNetParameters  ! Module containing weights and biases
		  IMPLICIT NONE
		  REAL(8), INTENT(IN) :: x(input_dim)
		  REAL(8), INTENT(IN) :: h_prev(num_layers, hidden_dim)
		  REAL(8), INTENT(IN) :: c_prev(num_layers, hidden_dim)
		  REAL(8), INTENT(OUT) :: h_t(num_layers, hidden_dim)
		  REAL(8), INTENT(OUT) :: c_t(num_layers, hidden_dim)
		  REAL(8), INTENT(OUT) :: output(output_dim)

		  ! Local variables
		  REAL(8), ALLOCATABLE :: x_t(:)
		  INTEGER :: layer

		  ! First layer input is the input vector x
		  ALLOCATE(x_t(input_dim))
		  x_t = x
		  
		  !Print*,'Size Weights L1'
		  !Print*,SIZE(weights_W_i_L1,1)
		  !Print*,SIZE(weights_W_i_L1,2)
		  !Print*,'Size h_prev'
		  !Print*,SIZE(h_prev,1)
		  !Print*,SIZE(h_prev,2)
		  !Print*,'Size c_prev'
		  !Print*,SIZE(c_prev,1)
		  !Print*,SIZE(c_prev,2)
		  
		  ! Layer 1
		  layer = 1
		  CALL lstm_cell(x_t, h_prev(layer, :), c_prev(layer, :), &
						 h_t(layer, :), c_t(layer, :), &
						 weights_W_i_L1(hidden_dim,input_dim), weights_U_i_L1(hidden_dim,hidden_dim), biases_W_i_L1, biases_U_i_L1, &
						 weights_W_f_L1(hidden_dim,input_dim), weights_U_f_L1(hidden_dim,hidden_dim), biases_W_f_L1, biases_U_f_L1, &
						 weights_W_c_L1(hidden_dim,input_dim), weights_U_c_L1(hidden_dim,hidden_dim), biases_W_c_L1, biases_U_c_L1, &
						 weights_W_o_L1(hidden_dim,input_dim), weights_U_o_L1(hidden_dim,hidden_dim), biases_W_o_L1, biases_U_o_L1, &
						 layer)
		  DEALLOCATE(x_t)
		  ! The input to the next layer is the hidden state from the current layer
		  ALLOCATE(x_t(hidden_dim))
		  x_t = h_t(layer, :)
			
		  !Print*,'h_t'
		  !Print*,h_t
		
		  ! Layer 2
		  layer = 2
		  CALL lstm_cell(x_t, h_prev(layer, :), c_prev(layer, :), &
						 h_t(layer, :), c_t(layer, :), &
						 weights_W_i_L2(hidden_dim,hidden_dim), weights_U_i_L2(hidden_dim,hidden_dim), biases_W_i_L2, biases_U_i_L2, &
						 weights_W_f_L2(hidden_dim,hidden_dim), weights_U_f_L2(hidden_dim,hidden_dim), biases_W_f_L2, biases_U_f_L2, &
						 weights_W_c_L2(hidden_dim,hidden_dim), weights_U_c_L2(hidden_dim,hidden_dim), biases_W_c_L2, biases_U_c_L2, &
						 weights_W_o_L2(hidden_dim,hidden_dim), weights_U_o_L2(hidden_dim,hidden_dim), biases_W_o_L2, biases_U_o_L2, &
						 layer)
		  DEALLOCATE(x_t)
		  ! The input to the next layer is the hidden state from the current layer
		  ALLOCATE(x_t(hidden_dim))
		  x_t = h_t(layer, :)

		  !Print*,'h_t'
		  !Print*,h_t

		  ! Layer 3 
		  layer = 3
		  CALL lstm_cell(x_t, h_prev(layer, :), c_prev(layer, :), &
						   h_t(layer, :), c_t(layer, :), &
						   weights_W_i_L3(hidden_dim,hidden_dim), weights_U_i_L3(hidden_dim,hidden_dim), biases_W_i_L3, biases_U_i_L3, &
						   weights_W_f_L3(hidden_dim,hidden_dim), weights_U_f_L3(hidden_dim,hidden_dim), biases_W_f_L3, biases_U_f_L3, &
						   weights_W_c_L3(hidden_dim,hidden_dim), weights_U_c_L3(hidden_dim,hidden_dim), biases_W_c_L3, biases_U_c_L3, &
						   weights_W_o_L3(hidden_dim,hidden_dim), weights_U_o_L3(hidden_dim,hidden_dim), biases_W_o_L3, biases_U_o_L3, &
						   layer)
		  DEALLOCATE(x_t)
		  !Print*,'h_t'
		  !Print*,h_t
		  ALLOCATE(x_t(hidden_dim))
		  x_t = h_t(layer, :)
		  
		  ! Compute the output using the final hidden state
		  output = MATMUL(weights_output, x_t) + biases_output
		  
		  DEALLOCATE(x_t)
		  
		  RETURN
		  
		END SUBROUTINE lstm_forward

		
		SUBROUTINE init_hidden(hidden_state, cell_state, hidden_dim)
			IMPLICIT NONE
			INTEGER, INTENT(IN) :: hidden_dim
			REAL(8), DIMENSION(hidden_dim), INTENT(OUT) :: hidden_state, cell_state

			hidden_state = 0.0D0
			cell_state = 0.0D0
		END SUBROUTINE init_hidden

		!Softplus Subroutine
		SUBROUTINE softplus_sub(x, sp_result, array_size)
			IMPLICIT NONE
			INTEGER :: i, array_size
			REAL*8, dimension(array_size), INTENT(IN) :: x         ! Input: array of values x
			REAL*8, dimension(array_size), INTENT(OUT) :: sp_result ! Output: array of softplus results

			! Softplus calculation: sp_result = log(1 + exp(x))
			do i = 1, size(x)
				sp_result(i) = log(1.0d0 + exp(x(i)))
			end do
		END SUBROUTINE softplus_sub
		
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