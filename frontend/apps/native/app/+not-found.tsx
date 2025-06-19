import { Link, Stack } from "expo-router";
import { Text, View } from "react-native";

export default function NotFoundScreen() {
  return (
    <>
      <Stack.Screen options={{ title: "Oops!" }} />
      <View className="flex-1 justify-center items-center p-6 bg-gray-50">
        <View className="items-center">
          <Text className="text-6xl mb-4">🤔</Text>
          <Text className="text-2xl font-bold text-gray-800 mb-2 text-center">
            Page Not Found
          </Text>
          <Text className="text-gray-500 text-center mb-8 max-w-sm">
            Sorry, the page you're looking for doesn't exist.
          </Text>
          <Link href="/" asChild>
            <View className="bg-blue-600 px-6 py-3 rounded-lg">
              <Text className="text-white font-medium">Go to Home</Text>
            </View>
          </Link>
        </View>
      </View>
    </>
  );
}