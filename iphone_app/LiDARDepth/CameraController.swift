/*
See LICENSE folder for this sample’s licensing information.

Abstract:
An object that configures and manages the capture pipeline to stream video and LiDAR depth data.
*/

import Foundation
import AVFoundation
import CoreImage
import UIKit
import SwiftUI


protocol CaptureDataReceiver: AnyObject {
    func onNewData(capturedData: CameraCapturedData)
    func onNewPhotoData(capturedData: CameraCapturedData)
}

class CameraController: NSObject, ObservableObject {
    
    enum ConfigurationError: Error {
        case lidarDeviceUnavailable
        case requiredFormatUnavailable
    }
    
    private let preferredWidthResolution = 1920
    
    private let videoQueue = DispatchQueue(label: "com.example.apple-samplecode.VideoQueue", qos: .userInteractive)
    
    private(set) var captureSession: AVCaptureSession!
    
    private var photoOutput: AVCapturePhotoOutput!
    private var depthDataOutput: AVCaptureDepthDataOutput!
    private var videoDataOutput: AVCaptureVideoDataOutput!
    private var outputVideoSync: AVCaptureDataOutputSynchronizer!
    
    private var textureCache: CVMetalTextureCache!
    
    weak var delegate: CaptureDataReceiver?
    
    var isFilteringEnabled = false {
        didSet {
            depthDataOutput.isFilteringEnabled = isFilteringEnabled
        }
    }
    
    private var folder_url : URL?
    private var frame_id : Int = 0
    private var can_write_data : Bool = false
    private var timestamps : String = ""
    private var start_timestamp : Double = 0
    private var data_start_frame_id : Int = -1
    private var end_collecting_data : Bool = false
    private var timestamp_exported : Bool = false
    
    override init() {
        
        // Create a texture cache to hold sample buffer textures.
        CVMetalTextureCacheCreate(kCFAllocatorDefault,
                                  nil,
                                  MetalEnvironment.shared.metalDevice,
                                  nil,
                                  &textureCache)
        
        super.init()
        
        // create a new folder to save data
        folder_url = createNewFolder()
        do {
            try setupSession()
        } catch {
            fatalError("Unable to configure the capture session.")
        }
    }
    
    func changeAppStatus(can_write_data : Bool, collecting_data: Bool){
        self.can_write_data = can_write_data
        if(collecting_data){
            if(can_write_data){ // when start collecting data
                data_start_frame_id = frame_id
            }
            else{ // when end collecting data
                end_collecting_data = true
            }
        }
    }
    
    private func createNewFolder() -> URL{
        let manager = FileManager.default
        let root_url = manager.urls(for: .documentDirectory, in: .userDomainMask).first
        let date = Date()
        let calendar = Calendar.current
        let year = calendar.component(.year, from: date)
        let month = calendar.component(.month, from: date)
        let day = calendar.component(.day, from: date)
        let hour = calendar.component(.hour, from: date)
        let minutes = calendar.component(.minute, from: date)
        let seconds = calendar.component(.second, from: date)
        let cur_date = String(format: "%04d", year) + "-" + String(format: "%02d", month) + "-" + String(format: "%02d", day) + "-" + String(format: "%02d", hour) + "-" + String(format: "%02d", minutes) + "-" + String(format: "%02d",seconds)
        let folder_url = root_url!.appendingPathComponent(cur_date)
        try? manager.createDirectory(at: folder_url, withIntermediateDirectories: true)
        // 2022-08-05-14-49-00
        return folder_url
    }
    
    private func setupSession() throws {
        captureSession = AVCaptureSession()
        captureSession.sessionPreset = .inputPriority

        // Configure the capture session.
        captureSession.beginConfiguration()
        
        try setupCaptureInput()
        setupCaptureOutputs()
        
        // Finalize capture session configuration.
        captureSession.commitConfiguration()
    }
    
    private func setupCaptureInput() throws {
        // Look up the LiDAR camera.
        guard let device = AVCaptureDevice.default(.builtInLiDARDepthCamera, for: .video, position: .back) else {
            throw ConfigurationError.lidarDeviceUnavailable
        }
        
        // Find a match that outputs video data in the format the app's custom Metal views require.
        guard let format = (device.formats.last { format in
            format.formatDescription.dimensions.width == preferredWidthResolution &&
            format.formatDescription.mediaSubType.rawValue == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange &&
            !format.isVideoBinned &&
            !format.supportedDepthDataFormats.isEmpty
        }) else {
            throw ConfigurationError.requiredFormatUnavailable
        }
        
        // Find a match that outputs depth data in the format the app's custom Metal views require.
        guard let depthFormat = (format.supportedDepthDataFormats.last { depthFormat in
            depthFormat.formatDescription.mediaSubType.rawValue == kCVPixelFormatType_DepthFloat16
        }) else {
            throw ConfigurationError.requiredFormatUnavailable
        }
        
        // Begin the device configuration.
        try device.lockForConfiguration()

        // Configure the device and depth formats.
        device.activeFormat = format
        device.activeDepthDataFormat = depthFormat

        // Finish the device configuration.
        device.unlockForConfiguration()
        
        print("Selected video format: \(device.activeFormat)")
        print("Selected depth format: \(String(describing: device.activeDepthDataFormat))")
        
        // Add a device input to the capture session.
        let deviceInput = try AVCaptureDeviceInput(device: device)
        captureSession.addInput(deviceInput)
    }
    
    private func setupCaptureOutputs() {
        // Create an object to output video sample buffers.
        videoDataOutput = AVCaptureVideoDataOutput()
        captureSession.addOutput(videoDataOutput)
        
        // Create an object to output depth data.
        depthDataOutput = AVCaptureDepthDataOutput()
        depthDataOutput.isFilteringEnabled = isFilteringEnabled
        captureSession.addOutput(depthDataOutput)

        // Create an object to synchronize the delivery of depth and video data.
        outputVideoSync = AVCaptureDataOutputSynchronizer(dataOutputs: [depthDataOutput, videoDataOutput])
        outputVideoSync.setDelegate(self, queue: videoQueue)

        // Enable camera intrinsics matrix delivery.
        guard let outputConnection = videoDataOutput.connection(with: .video) else { return }
        if outputConnection.isCameraIntrinsicMatrixDeliverySupported {
            outputConnection.isCameraIntrinsicMatrixDeliveryEnabled = true
        }
        
        // Create an object to output photos.
        photoOutput = AVCapturePhotoOutput()
        photoOutput.maxPhotoQualityPrioritization = .quality
        captureSession.addOutput(photoOutput)

        // Enable delivery of depth data after adding the output to the capture session.
        photoOutput.isDepthDataDeliveryEnabled = true
    }
    
    func startStream() {
        captureSession.startRunning()
    }
    
    func stopStream() {
        captureSession.stopRunning()
    }
    
    func hasZero(from pixelBuffer: CVPixelBuffer) -> Bool {
        CVPixelBufferLockBaseAddress(pixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
        
        let rows = CVPixelBufferGetHeight(pixelBuffer)
        let cols = CVPixelBufferGetWidth(pixelBuffer)
        
        let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer)
        let buffer = baseAddress?.assumingMemoryBound(to: UInt16.self)
        
        
        // Get the pixel.  You could iterate here of course to get multiple pixels!
        
        var out = false
        
        for y in 1...rows{
            for x in 1...cols{
                let baseAddressIndex = y  * cols + x
                let pixel = buffer![baseAddressIndex]
                if (pixel == 0) {
                    out = true
                    break
                }
            }
            if (out) {
                break
            }
        }

        CVPixelBufferUnlockBaseAddress(pixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
        
        return out
    }
    
    func depthCVPixelToData(from pixelBuffer: CVPixelBuffer) -> CVPixelBuffer{
        
        CVPixelBufferLockBaseAddress(pixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
        
        let rows = CVPixelBufferGetHeight(pixelBuffer)
        let cols = CVPixelBufferGetWidth(pixelBuffer)

        let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer)
        let buffer = baseAddress?.assumingMemoryBound(to: Float16.self)
        
        var new_cvbuffer : CVPixelBuffer? = nil
        
        CVPixelBufferCreate(kCFAllocatorDefault, cols, rows, kCVPixelFormatType_16Gray, nil, &new_cvbuffer)
        
        CVPixelBufferLockBaseAddress(new_cvbuffer!, CVPixelBufferLockFlags(rawValue: 0))
        
        let outBaseAddress = CVPixelBufferGetBaseAddress(new_cvbuffer!)
        
        let uint16buffer = outBaseAddress?.assumingMemoryBound(to: UInt16.self)
        
        for y in 0...(rows-1){
            for x in 0...(cols-1){
                let baseAddressIndex = y  * cols + x
                var pixel = buffer![baseAddressIndex]
                if (pixel.isNaN || pixel.isInfinite) {
                    pixel = 0
                }
                let pixel_uint16 : UInt16 = UInt16(pixel * 1000)
                uint16buffer![baseAddressIndex] = pixel_uint16
            }
        }
        
        CVPixelBufferUnlockBaseAddress(pixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
        CVPixelBufferUnlockBaseAddress(new_cvbuffer!, CVPixelBufferLockFlags(rawValue: 0))
        return new_cvbuffer!
    }
    
    func save_depth_data(from depthDataMap : CVPixelBuffer, url: URL){
        let new_cvbuffer : CVPixelBuffer = depthCVPixelToData(from: depthDataMap)
        CVPixelBufferLockBaseAddress(new_cvbuffer, CVPixelBufferLockFlags(rawValue: 0))
        let height = CVPixelBufferGetHeight(new_cvbuffer)
        let yBaseAddress = CVPixelBufferGetBaseAddress(new_cvbuffer)
        let yBytesPerRow = CVPixelBufferGetBytesPerRow(new_cvbuffer)
        CVPixelBufferUnlockBaseAddress(new_cvbuffer, CVPixelBufferLockFlags(rawValue: 0))
        let yLength = yBytesPerRow *  height
        let d_data = Data(bytes: yBaseAddress!, count: yLength)
        try? d_data.write(to: url)
    }
    
    func save_timestamp_data(from timestamp: CMTime){
        // save timestamps, write later as one single file
        if(start_timestamp == 0){
            start_timestamp = timestamp.seconds
            print("start timestamp: ", start_timestamp)
        }
        self.timestamps += String(timestamp.seconds - start_timestamp) + ","
    }
    
    func matrix_to_string(from matrix : matrix_float3x3, name : String) -> String{
        var matrix_name = name
        matrix_name += "["
        for i in 0...2{
            matrix_name += "["
            for j in 0...2{
                matrix_name += String(matrix[i][j]) + ","
            }
            matrix_name = String(matrix_name.dropLast())
            matrix_name += "],\n"
        }
        matrix_name = String(matrix_name.dropLast(2))
        matrix_name += "]"
        return matrix_name
    }
    
    func matrix_to_string(from matrix : matrix_float4x3, name : String) -> String{
        var matrix_name = name
        matrix_name += "["
        for i in 0...3{
            matrix_name += "["
            for j in 0...2{
                matrix_name += String(matrix[i][j]) + ","
            }
            matrix_name = String(matrix_name.dropLast())
            matrix_name += "],\n"
        }
        matrix_name = String(matrix_name.dropLast(2))
        matrix_name += "]"
        return matrix_name
    }
    
    func save_camera_calib_data(from cameraCalibrationData: AVCameraCalibrationData, calibration_url: URL, inverse_lookup_table_url: URL){
        
//        var cameraIntrinsic : String = "Camera Intrinsic: \n"
//        cameraIntrinsic += "["
//        for i in 0...2{
//            cameraIntrinsic += "["
//            for j in 0...2{
//                cameraIntrinsic += String(cameraCalibrationData.intrinsicMatrix[i][j]) + ","
//            }
//            cameraIntrinsic = String(cameraIntrinsic.dropLast())
//            cameraIntrinsic += "],\n"
//        }
//        cameraIntrinsic = String(cameraIntrinsic.dropLast(2))
//        cameraIntrinsic += "]"

        let cameraIntrinsic : String = matrix_to_string(from: cameraCalibrationData.intrinsicMatrix, name: "Camera Intrinsic: \n")
        let cameraExtrinsic : String = matrix_to_string(from: cameraCalibrationData.extrinsicMatrix, name: "Camera Extrinsic: \n")
        let lensDistortionCenter : String = "Distortion Center: \n" +  String(Float(cameraCalibrationData.lensDistortionCenter.x)) + "," + String(Float(cameraCalibrationData.lensDistortionCenter.y))
        let calibrationData : String = cameraIntrinsic + "\n" + cameraExtrinsic + "\n" + lensDistortionCenter

        try? calibrationData.write(toFile: calibration_url.path, atomically: false, encoding: String.Encoding.utf8)
        try? cameraCalibrationData.inverseLensDistortionLookupTable!.write(to: inverse_lookup_table_url)
    }
    
    func save_data(sampleBuffer : CMSampleBuffer, depthDataMap : CVPixelBuffer, timestamp: CMTime, cameraCalibrationData:AVCameraCalibrationData){
        // retrieve rgb data
        let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer)!
        let rgb_ciImage = CIImage(cvPixelBuffer: imageBuffer)
        let rgb_cgImage = CIContext(options: nil).createCGImage(rgb_ciImage, from: rgb_ciImage.extent)!
        let rgb_img = UIImage(cgImage: rgb_cgImage)
        
        if let img = rgb_img.jpegData(compressionQuality: 0.3) {
            let rgb_url = folder_url!.appendingPathComponent(String(frame_id) + ".jpeg")
            let depth_url = folder_url!.appendingPathComponent(String(frame_id) + ".bin")
            let calibration_url = folder_url!.appendingPathComponent(String(frame_id) + "_calibration.txt")
            let inverse_lookup_table_url = folder_url!.appendingPathComponent(String(frame_id) + "_distortion_table.bin")
            
            try! img.write(to: rgb_url)
            save_depth_data(from: depthDataMap, url: depth_url)
            save_timestamp_data(from: timestamp)
            save_camera_calib_data(from: cameraCalibrationData, calibration_url: calibration_url, inverse_lookup_table_url: inverse_lookup_table_url)
            frame_id += 1
        }
    }
    
    private func exportTimeStampData(){
        let csv_url = folder_url!.appendingPathComponent("timestamps.csv")
        timestamps += String(data_start_frame_id) // add id to the end of the array
        try? timestamps.write(toFile: csv_url.path, atomically: false, encoding: String.Encoding.utf8)
    }
    
    
//    func lerp(x1: (CGFloat, CGFloat, CGFloat), x2: (CGFloat, CGFloat, CGFloat), ratio: CGFloat) -> (CGFloat, CGFloat, CGFloat){
//        let r = x1.0 + (x2.0 - x1.0) *  ratio
//        let g = x1.1 + (x2.1 - x1.1) *  ratio
//        let b = x1.2 + (x2.2 - x1.2) *  ratio
//        return (r, g, b)
//    }
//
//    func bilinear_interpolation(data: UnsafePointer<UInt8>, width: Int, height: Int, point: CGPoint, tl: CGPoint, tr: CGPoint, br: CGPoint, bl: CGPoint) -> (CGFloat, CGFloat, CGFloat){
//        let tl_RGB = get_RGB_from_CGpoint(data: data, width: width, point: tl)
//        let tr_RGB = get_RGB_from_CGpoint(data: data, width: width, point: tr)
//        let br_RGB = get_RGB_from_CGpoint(data: data, width: width, point: br)
//        let bl_RGB = get_RGB_from_CGpoint(data: data, width: width, point: bl)
//        let x_ratio = point.x.truncatingRemainder(dividingBy: 1)
//        let y_ratio = point.y.truncatingRemainder(dividingBy: 1)
//        let interp_y1 = lerp(x1: tl_RGB, x2: tr_RGB, ratio: x_ratio)
//        let interp_y2 = lerp(x1: bl_RGB, x2: br_RGB, ratio: x_ratio)
//        let interp_RGB = lerp(x1: interp_y1, x2: interp_y2, ratio: y_ratio)
//        return interp_RGB
//    }
//
//    func get_RGB_from_CGpoint(data: UnsafePointer<UInt8>, width: Int, point: CGPoint) -> (CGFloat, CGFloat, CGFloat){
//        let pixelData = (width * Int(point.y) + Int(point.x)) * 3
//        let point_RGB = (CGFloat(data[pixelData]) / 255.0, CGFloat(data[pixelData + 1]) / 255.0, CGFloat(data[pixelData + 2]) / 255.0)
//        return point_RGB
//    }
//
//    func get_RGB_from_CGpoint_new(data: UnsafePointer<UInt8>, width: Int, point: CGPoint) -> (CGFloat, CGFloat, CGFloat){
//        let pixelData = (width * Int(point.y) + Int(point.x)) * 3
//        let point_RGB = (CGFloat(data[pixelData]) / 255.0, CGFloat(data[pixelData + 1]) / 255.0, CGFloat(data[pixelData + 2]) / 255.0)
//        return point_RGB
//    }
//
//
//    func apply_distortion_correction(sampleBuffer : CMSampleBuffer, lookupTable: Data, opticalCenter: CGPoint) {
//        let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer)!
//        let rgb_ciImage = CIImage(cvPixelBuffer: imageBuffer)
//        let rgb_cgImage = CIContext(options: nil).createCGImage(rgb_ciImage, from: rgb_ciImage.extent)!
//
//        print("new image!")
//
//        //  print(CVPixelBufferGetPixelFormatType(imageBuffer)) //kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
//
//        let cf_data = rgb_cgImage.dataProvider?.data!
//        let data = CFDataGetBytePtr(cf_data)!
//
//        var new_cvbuffer : CVPixelBuffer? = nil
//        CVPixelBufferCreate(kCFAllocatorDefault, rgb_cgImage.width, rgb_cgImage.height, kCVPixelFormatType_24RGB, nil, &new_cvbuffer)
//
//        CVPixelBufferLockBaseAddress(new_cvbuffer!, CVPixelBufferLockFlags(rawValue: 0))
//
//        let outBaseAddress = CVPixelBufferGetBaseAddress(new_cvbuffer!)
//
//        let rgb_out_ptr = outBaseAddress?.assumingMemoryBound(to: UInt8.self)
//
//
//        for x in 0 ... (rgb_cgImage.width-1){
//            for y in 0 ... (rgb_cgImage.height-1) {
//                let undistorted_point = CGPoint(x: x,y: y)
//                let distorted_point = lensDistortionPoint(for: undistorted_point, lookupTable: lookupTable, distortionOpticalCenter: opticalCenter, imageSize: CGSize(width: rgb_cgImage.width, height: rgb_cgImage.height))
//                // bilinear interpolation to get value
//                let tl = CGPoint(x: Int(distorted_point.x), y: Int(distorted_point.y))
//                let tr =  CGPoint(x: Int(distorted_point.x) + 1, y: Int(distorted_point.y))
//                let br = CGPoint(x: Int(distorted_point.x) + 1, y: Int(distorted_point.y) + 1)
//                let bl = CGPoint(x: Int(distorted_point.x), y: Int(distorted_point.y) + 1)
//                let undistorted_RGB = bilinear_interpolation(data: data, width: rgb_cgImage.width, height: rgb_cgImage.height, point: distorted_point, tl: tl, tr: tr, br: br, bl: bl)
//
//                let rgb = get_RGB_from_CGpoint(data: data, width: rgb_cgImage.width, point: undistorted_point)
//
//                let baseAddressIndex = (x * rgb_cgImage.height + y) * 3
//                rgb_out_ptr![baseAddressIndex] = UInt8(rgb.0 * 255)
//                rgb_out_ptr![baseAddressIndex + 1] = UInt8(rgb.1 * 255)
//                rgb_out_ptr![baseAddressIndex + 2] = UInt8(rgb.2 * 255)
//            }
//        }
//    }
//
//    func lensDistortionPoint(for point: CGPoint, lookupTable: Data, distortionOpticalCenter opticalCenter: CGPoint, imageSize: CGSize) -> CGPoint {
//        // The lookup table holds the relative radial magnification for n linearly spaced radii.
//        // The first position corresponds to radius = 0
//        // The last position corresponds to the largest radius found in the image.
//
//        // Determine the maximum radius.
//        let delta_ocx_max = Float(max(opticalCenter.x, imageSize.width  - opticalCenter.x))
//        let delta_ocy_max = Float(max(opticalCenter.y, imageSize.height - opticalCenter.y))
//        let r_max = sqrt(delta_ocx_max * delta_ocx_max + delta_ocy_max * delta_ocy_max)
//
//        // Determine the vector from the optical center to the given point.
//        let v_point_x = Float(point.x - opticalCenter.x)
//        let v_point_y = Float(point.y - opticalCenter.y)
//
//        // Determine the radius of the given point.
//        let r_point = sqrt(v_point_x * v_point_x + v_point_y * v_point_y)
//
//        // Look up the relative radial magnification to apply in the provided lookup table
//        let magnification: Float = lookupTable.withUnsafeBytes { (lookupTableValues: UnsafePointer<Float>) in
//            let lookupTableCount = lookupTable.count / MemoryLayout<Float>.size
//
//            if r_point < r_max {
//                // Linear interpolation
//                let val   = r_point * Float(lookupTableCount - 1) / r_max
//                let idx   = Int(val)
//                let frac  = val - Float(idx)
//
//                let mag_1 = lookupTableValues[idx]
//                let mag_2 = lookupTableValues[idx + 1]
//
//                return (1.0 - frac) * mag_1 + frac * mag_2
//            } else {
//                return lookupTableValues[lookupTableCount - 1]
//            }
//        }
//
//        // Apply radial magnification
//        let new_v_point_x = v_point_x + magnification * v_point_x
//        let new_v_point_y = v_point_y + magnification * v_point_y
//
//        // Construct output
//        return CGPoint(x: opticalCenter.x + CGFloat(new_v_point_x), y: opticalCenter.y + CGFloat(new_v_point_y))
//    }
    
}

// MARK: Output Synchronizer Delegate
extension CameraController: AVCaptureDataOutputSynchronizerDelegate {
    
    func dataOutputSynchronizer(_ synchronizer: AVCaptureDataOutputSynchronizer,
                                didOutput synchronizedDataCollection: AVCaptureSynchronizedDataCollection) {
        // Retrieve the synchronized depth and sample buffer container objects.
        guard let syncedDepthData = synchronizedDataCollection.synchronizedData(for: depthDataOutput) as? AVCaptureSynchronizedDepthData,
              let syncedVideoData = synchronizedDataCollection.synchronizedData(for: videoDataOutput) as? AVCaptureSynchronizedSampleBufferData else { return }
        
        guard let pixelBuffer = syncedVideoData.sampleBuffer.imageBuffer,
              let cameraCalibrationData = syncedDepthData.depthData.cameraCalibrationData else { return }
        

        // Package the captured data.
        let data = CameraCapturedData(depth: syncedDepthData.depthData.depthDataMap.texture(withFormat: .r16Float, planeIndex: 0, addToCache: textureCache),
                                      colorY: pixelBuffer.texture(withFormat: .r8Unorm, planeIndex: 0, addToCache: textureCache),
                                      colorCbCr: pixelBuffer.texture(withFormat: .rg8Unorm, planeIndex: 1, addToCache: textureCache),
                                      cameraIntrinsics: cameraCalibrationData.intrinsicMatrix,
                                      cameraReferenceDimensions: cameraCalibrationData.intrinsicMatrixReferenceDimensions)
        
        // apply distortion correction to both rgb images and depth images
//        apply_distortion_correction(sampleBuffer : syncedVideoData.sampleBuffer, lookupTable: cameraCalibrationData.lensDistortionLookupTable!, opticalCenter: cameraCalibrationData.lensDistortionCenter)
        
        
        // save to local directory
        if(can_write_data){
            save_data(sampleBuffer: syncedVideoData.sampleBuffer, depthDataMap: syncedDepthData.depthData.depthDataMap, timestamp: syncedVideoData.timestamp, cameraCalibrationData: cameraCalibrationData)
        }
        
        if(end_collecting_data && !timestamp_exported){
            exportTimeStampData()
            timestamp_exported = true
        }
        
        // delegate data
        delegate?.onNewData(capturedData: data)
    }
    
}

// MARK: Photo Capture Delegate
extension CameraController: AVCapturePhotoCaptureDelegate {
    
    func capturePhoto() {
        var photoSettings: AVCapturePhotoSettings
        if  photoOutput.availablePhotoPixelFormatTypes.contains(kCVPixelFormatType_420YpCbCr8BiPlanarFullRange) {
            photoSettings = AVCapturePhotoSettings(format: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
            ])
        } else {
            photoSettings = AVCapturePhotoSettings()
        }
        
        // Capture depth data with this photo capture.
        photoSettings.isDepthDataDeliveryEnabled = true
        photoOutput.capturePhoto(with: photoSettings, delegate: self)
    }
    
    func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        
        // Retrieve the image and depth data.
        guard let pixelBuffer = photo.pixelBuffer,
              let depthData = photo.depthData,
              let cameraCalibrationData = depthData.cameraCalibrationData else { return }
        
        // Stop the stream until the user returns to streaming mode.
        stopStream()
        
        // Convert the depth data to the expected format.
        let convertedDepth = depthData.converting(toDepthDataType: kCVPixelFormatType_DepthFloat16)
        
        // Package the captured data.
        let data = CameraCapturedData(depth: convertedDepth.depthDataMap.texture(withFormat: .r16Float, planeIndex: 0, addToCache: textureCache),
                                      colorY: pixelBuffer.texture(withFormat: .r8Unorm, planeIndex: 0, addToCache: textureCache),
                                      colorCbCr: pixelBuffer.texture(withFormat: .rg8Unorm, planeIndex: 1, addToCache: textureCache),
                                      cameraIntrinsics: cameraCalibrationData.intrinsicMatrix,
                                      cameraReferenceDimensions: cameraCalibrationData.intrinsicMatrixReferenceDimensions)
        delegate?.onNewPhotoData(capturedData: data)
    }
}
