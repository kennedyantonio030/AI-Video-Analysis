const { parentPort } = require("worker_threads");
const path = require("path");
const sharp = require("sharp");
const { ssim } = require("ssim.js");

parentPort.on("message", async (pairs) => {
  const ssimResults = [];

  for (let i = 0; i < pairs.length; i++) {
    const path1 = path.join("/tmp/frames", pairs[i][0]);
    const path2 = path.join("/tmp/frames", pairs[i][1]);
    const img1 = await sharp(path1).raw().ensureAlpha().toBuffer();
    const img2 = await sharp(path2).raw().ensureAlpha().toBuffer();

    const metadata1 = await sharp(path1).metadata();

    const image1Data = {
      width: metadata1.width,
      height: metadata1.height,
      data: img1,
    };

    const image2Data = {
      width: metadata1.width,
      height: metadata1.height,
      data: img2,
    };
    ssimResults.push(ssim(image1Data, image2Data).mssim);
  }

  parentPort.postMessage(ssimResults);
});
