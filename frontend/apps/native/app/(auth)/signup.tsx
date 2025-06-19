import { useRouter } from "expo-router";
import React, { useState } from "react";
import { View, Text, TextInput, TouchableOpacity, Alert, ActivityIndicator } from "react-native";

export default function SignUpScreen() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const router = useRouter();
  const [loading, setLoading] = useState(false);

  const handleSignUp = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${process.env.EXPO_PUBLIC_SERVER_URL}/api/auth/signup`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });

      if (response.ok) {
        const { error } = await authClient.signIn.email({ email, password });
        if (error) {
          Alert.alert(
            "Sign Up Successful",
            "Your account was created, but we couldn't log you in automatically. Please log in manually."
          );
          router.replace("/(auth)/login");
        } else {
          Alert.alert("Welcome!", "Your account has been created successfully.");
          router.back();
          router.replace("/(tabs)/profile");
        }
      } else {
        const errorData = await response.json();
        Alert.alert("Sign Up Failed", errorData.message || "An unexpected error occurred.");
      }
    } catch (error: unknown) {
      const errorMessage =
        error instanceof Error ? error.message : "A network error occurred.";
      Alert.alert("Sign Up Failed", errorMessage);
    } finally {
      setLoading(false);
    }
  };

  return (
    <View className="flex-1 bg-gray-100 p-6 justify-center">
      <Text className="text-3xl font-bold text-center mb-8 text-gray-800">
        Create Account
      </Text>
      <View className="space-y-4">
        <TextInput
          className="bg-white border border-gray-300 rounded-lg px-4 py-3 text-base"
          placeholder="Email"
          value={email}
          onChangeText={setEmail}
          keyboardType="email-address"
          autoCapitalize="none"
        />
        <TextInput
          className="bg-white border border-gray-300 rounded-lg px-4 py-3 text-base"
          placeholder="Password"
          value={password}
          onChangeText={setPassword}
          secureTextEntry
        />
        <TouchableOpacity
          onPress={handleSignUp}
          className="bg-blue-600 rounded-lg p-4 items-center justify-center"
          disabled={loading}
        >
          {loading ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <Text className="text-white font-bold text-base">Sign Up</Text>
          )}
        </TouchableOpacity>
      </View>
      <TouchableOpacity
        onPress={() => router.replace("/(auth)/login")}
        className="mt-6"
      >
        <Text className="text-center text-blue-600">
          Already have an account? Login
        </Text>
      </TouchableOpacity>
    </View>
  );
}